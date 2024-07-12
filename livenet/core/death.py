import math
import collections

from ai_libs.simple_log import LOGD


class LivenessObserver:
    @staticmethod
    def sign(x):
        return 1 if x >= 0 else -1

    def __init__(self, context):
        self.threshold = 0.05
        self.context = context
        self.n_small = 0
        self.last_sign = 0
        n_sign = context.liveness_die_after_n_sign_changes
        # allow other n_sign times for value move until settle in
        self.sign_history = collections.deque(2 * n_sign * [-math.inf])

    def put(self, x: float):
        sign = self.sign(x)
        if sign != self.last_sign:
            self.sign_history.popleft()
            self.sign_history.append(self.context.tick)
        self.last_sign = sign
        if abs(x) <= self.threshold:
            self.n_small += 1
        else:
            self.n_small = 0

    # 0 - ok, -1 - die, 1 - would die, but at least one history value is above threshold
    def status(self):
        # return -1
        if self.sign_history[0] == -math.inf:
            return 0
        n_sign = self.context.liveness_die_after_n_sign_changes
        history_len = 1 + self.context.tick - self.sign_history[len(self.sign_history) - n_sign + 1]
        if self.n_small < history_len:
            LOGD(f"Would die but at least one history value is above threshold "
                f"n_small={self.n_small} history_len={history_len}")
            return 1
        return -1


