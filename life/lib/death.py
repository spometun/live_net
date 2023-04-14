import math
import dataclasses

from life.lib.simple_log import LOG


class LivenessObserver:
    def __init__(self):
        self.dead = False
        self.threshold = 0.01
        self.weight = 0.05
        self.value = math.pow(1 - self.weight, -100)

    def put(self, x: float):
        self.value = (1 - self.weight) * self.value + self.weight * x

    def looks_ok(self):
        return math.fabs(self.value) >= self.threshold


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
