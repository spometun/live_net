import math
import dataclasses
import collections

import numpy as np

from life.lib.simple_log import LOG


class LivenessObserver:
    @staticmethod
    def sign(x):
        return 1 if x >= 0 else -1

    def __init__(self, context, value):
        self.threshold = 0.05
        self.n_sign_history = 5
        self.dead = False
        self.context = context
        self.n_small = 0
        self.value = value
        self.last_sign = self.sign(value)
        # allow other n_sign_history for value move until settle in
        self.sign_history = collections.deque(2 * self.n_sign_history * [-math.inf])

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
        if self.sign_history[0] == -math.inf:
            return 0
        history_len = 1 + self.context.tick - self.sign_history[len(self.sign_history) - self.n_sign_history + 1]
        if self.n_small < history_len:
            LOG(f"Would die but but at least one history value is above threshold "
                f"n_small={self.n_small} history_len={history_len}")
            return 1
        return -1


class DeathStat:
    def __init__(self):
        self.dangle_neurons: int = 0

    def on_dangle_neuron(self, dangle: "DestinationNeuron"):
        self.dangle_neurons += 1
        LOG(f"{dangle.name} became dangle, total dangle = {self.dangle_neurons}")

    def off_dangle_neuron(self, dangle: "DestinationNeuron"):
        self.dangle_neurons -= 1
        LOG(f"dangle {dangle.name} died, total dangle = {self.dangle_neurons}")
        assert self.dangle_neurons >= 0, "Internal error"
