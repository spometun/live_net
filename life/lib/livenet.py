import typing
from typing import List, Self
from overrides import override
import numpy as np
import pytest
import abc
import torch
import torch.nn as nn
import random
import pickle
from life.lib.graph import GraphNode, NodesHolder
import life.lib.utils as utils


from life.lib.simple_log import LOG
import life.lib.optimizer as optimizer


def test_dev():
    print("O merciful God")


class Neuron(GraphNode):
    def __init__(self, context: "Context"):
        super().__init__()
        self.context = context
        self._output = None

    @typing.final
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
    def __init__(self, context: "Context"):
        super().__init__(context)
        self.axons: List[Synapse] = []

    def connect_to(self, destination: "DestinationNeuron"):
        synapse = Synapse(self, destination)
        return synapse

    @override
    def get_adjacent_nodes(self) -> List[GraphNode]:
        return []


class DestinationNeuron(Neuron):
    def __init__(self, context: "Context"):
        super().__init__(context)
        self.dendrites: List[Synapse] = []
        self.b = context.obtain_float_parameter("n")
        self.optimizer = optimizer.SGD1(self.b)

    def _compute_output(self) -> torch.Tensor:
        if len(self.dendrites) == 0:
            output = self.b
        else:
            output = self.dendrites[0].output()
            for synapse in self.dendrites[1:]:
                output = output + synapse.output()
            output = output + self.b
        output = torch.relu(output)
        return output

    def connect_from(self, source: SourceNeuron):
        synapse = Synapse(source, self)
        return synapse

    @override
    def get_adjacent_nodes(self) -> List[GraphNode]:
        return self.dendrites


class DataNeuron(SourceNeuron):
    def __init__(self, context: "Context"):
        super().__init__(context)

    def set_output(self, value: torch.Tensor):
        assert len(value.shape) == 1
        self._output = value

    def _compute_output(self) -> torch.Tensor:
        raise RuntimeError("Should never be called")


class RegularNeuron(DestinationNeuron, SourceNeuron):
    def __init__(self, context: "Context"):
        super().__init__(context)


class Synapse(GraphNode):
    def __init__(self, source: SourceNeuron, destination: DestinationNeuron):
        assert source != destination
        self.source = source
        self.destination = destination
        assert source not in (synapse.source for synapse in destination.dendrites), "Connection already exists"
        destination.dendrites.append(self)
        assert destination not in (synapse.destination for synapse in source.axons), "Connection already exists"
        source.axons.append(self)
        self.k = source.context.obtain_float_parameter("s")
        with torch.no_grad():
            self.k[...] = torch.tensor(random.random())
        self.optimizer = optimizer.SGD1(self.k)

    def output(self):
        output = self.k * self.source.compute_output()
        return output

    def get_adjacent_nodes(self) -> List[GraphNode]:
        return [self.source]


class Context:
    def __init__(self, module: nn.Module):
        self.module = module
        self._n_params = 0
        self._name_counters = {}

    def obtain_float_parameter(self, name_prefix: str) -> nn.Parameter:
        if name_prefix not in self._name_counters:
            self._name_counters[name_prefix] = -1
        self._name_counters[name_prefix] += 1
        name = f"{name_prefix}{self._name_counters[name_prefix]}"
        self._n_params += 1
        param = nn.Parameter(torch.tensor(0.0))
        self.module.register_parameter(name, param)
        return param


class LiveNet(nn.Module):
    def __init__(self, n_inputs, n_middle, n_outputs):
        super().__init__()
        self.context = Context(self)
        self.inputs = [DataNeuron(self.context) for _ in range(n_inputs)]
        self.outputs = [DestinationNeuron(self.context) for _ in range(n_outputs)]
        self.root = NodesHolder(self.outputs)
        '''
        for input_ in self.inputs:
            for output in self.outputs:
                input_.connect_to(output)
                '''
        for i in range(n_middle):
            neuron = RegularNeuron(self.context)
            for input_ in self.inputs:
                input_.connect_to(neuron)
            for output in self.outputs:
                output.connect_from(neuron)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2, "Invalid input shape"
        # assert x.shape[1] == len(self.inputs), "Invalid input dimension"
        '''
        s1 = self.get_parameter("s1")
        s2 = self.get_parameter("s2")
        n0 = self.get_parameter("n0")
        y = torch.zeros(x.shape[0], len(self.outputs))
        y[:, 0] = nn.functional.relu(x[:, 0] * s1 + x[:, 1] * s2 + n0)
        return y
'''
        self.root.visit("clear_output")
        for i in range(x.shape[1]):
            self.inputs[i].set_output(x[:, i])
        y = torch.zeros(x.shape[0], len(self.outputs))
        for i, output in enumerate(self.outputs):
            y[:, i] = output.compute_output()
        return y


def export_onnx(model: nn.Module, dummy_input):
    torch.onnx.export(model, dummy_input, "/home/spometun/model.onnx", verbose=False)


if __name__ == "__main__":
    utils.set_seed()
    x = torch.tensor([[0., 0], [0, 1], [1, 0], [1, 1]])
    net = LiveNet(2, 2, 1)
    net.forward(x)
    net.forward(x)

    LOG('a')
    export_onnx(net, x)
    LOG('b')

    p = [p for p in net.named_parameters()]
    s1 = net.get_parameter("s1")

    LOG("Here")
    # net = LiveNet(2, 1)
