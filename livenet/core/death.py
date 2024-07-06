import math
import dataclasses
import collections

import numpy as np

from ai_libs.simple_log import LOG, LOGD


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
        if x <= self.threshold:
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


class HealthStat:
    def __init__(self):
        self.dangle_neurons: int = 0
        self.dangle_destination_neurons: int = 0
        self.useless_neurons: int = 0
        self.useless_data_neurons: int = 0
        self.dangle = set()
        self.useless = set()

    def on_dangle_neuron(self, dangle: "DestinationNeuron"):
        LOG(f"{dangle.name} became dangle")
        assert dangle not in self.dangle
        self.dangle.add(dangle)

    def off_dangle_neuron(self, dangle: "DestinationNeuron"):
        LOGD(f"{dangle.name} is not dangle any more")
        self.dangle.remove(dangle)

    def on_useless_neuron(self, useless: "SourceNeuron"):
        LOG(f"{useless.name} became useless")
        assert useless not in self.useless
        self.useless.add(useless)

    def off_useless_neuron(self, useless: "SourceNeuron"):
        LOGD(f"{useless.name} is not useless any more")
        self.useless.remove(useless)

    def get_stat(self) -> dict:
        stat = {
            "dangle": {"RegularNeuron": 0, "DestinationNeuron": 0},
            "useless": {"RegularNeuron": 0, "DataNeuron": 0}
        }
        for neuron in self.dangle:
            class_name = neuron.__class__.__name__
            stat["dangle"][class_name] += 1
        for neuron in self.useless:
            class_name = neuron.__class__.__name__
            stat["useless"][class_name] += 1
        return stat
