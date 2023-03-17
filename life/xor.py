from typing import List, Self
import numpy as np
import pytest
import abc
import torch
import torch.nn as nn
import random

import lib


def test_dev():
    print("O merciful God")


class Neuron:
    def __init__(self):
        self._output = None

    def compute_output(self) -> torch.Tensor:
        if self._output is None:
            self._output = self._compute_output()
        return self._output

    def clear_output(self):
        self._output = None

    @abc.abstractmethod
    def _compute_output(self) -> torch.Tensor: ...


# noinspection PyAbstractClass
class SourceNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.axons: List[Synapse] = []

    def connect_to(self, destination):
        synapse = Synapse(self, destination)
        return synapse


# noinspection PyAbstractClass
class DestinationNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.dendrites: List[Synapse] = []

    def connect_from(self, source):
        synapse = Synapse(source, self)
        return synapse


class DataNeuron(SourceNeuron):
    def __init__(self):
        super().__init__()

    def set_output(self, value: torch.Tensor):
        assert len(value.shape) == 2
        self._output = value

    def _compute_output(self) -> torch.Tensor:
        raise NotImplementedError("Should never be called")


class Synapse:
    def __init__(self, source: SourceNeuron, destination: DestinationNeuron):
        self.source = source
        self.destination = destination
        assert source not in (synapse.source for synapse in destination.dendrites), "Connection already exists"
        destination.dendrites.append(self)
        assert destination not in (synapse.destination for synapse in source.axons), "Connection already exists"
        source.axons.append(self)
        self.k = torch.tensor(random.random())

    def output(self):
        output = self.k * self.source.compute_output()
        return output


class RegularNeuron(SourceNeuron, DestinationNeuron):
    def __init__(self):
        super().__init__()
        self.b = torch.tensor(0.)
        self.output = None

    def compute_output(self) -> torch.Tensor:
        if self.output is None:
            self.output = self.b
            for synapse in self.dendrites:
                self.output = self.output + synapse.output()
            self.output = torch.nn.functional.relu(self.output)
        return self.output

    def clear_output(self):
        self.output = None


class LiveNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.inputs = [InputNeuron() for i in range(n_inputs)]
        self.outputs = [RegularNeuron() for i in range(n_outputs)]
        pass

    def forward(self, x):
        assert len(x.shape) == 2, "Invalid input shape"
        assert x.shape[1] == len(self.inputs), "Invalid input dimension"
        pass


if __name__ == "__main__":
    lib.utils.set_seed()
    src = DataNeuron()
    dst = RegularNeuron()
    src.connect_to(dst)
    print("Here")
    # net = LiveNet(2, 1)
