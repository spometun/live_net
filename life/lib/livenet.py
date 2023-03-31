import typing
from typing import List, Self, Union
from overrides import override
import numpy as np
import pytest
import abc
import torch
import torch.nn as nn
import pickle
import math
import random
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
    def __init__(self, context: "Context", activation):
        super().__init__(context)
        self.dendrites: List[Synapse] = []
        self.b = context.obtain_float_parameter("n")
        self.optimizer = optimizer.SGD1(self.b)
        self.activation = activation

    def on_grad_update(self):
        self.optimizer.step()

    def _compute_output(self) -> torch.Tensor:
        if len(self.dendrites) == 0:
            output = self.b
        else:
            if self.context.reduce_sum_computation:
                outputs = []
                for synapse in self.dendrites:
                    outputs.append(synapse.output())
                all_ = torch.cat(outputs, 1)
                output = torch.sum(all_, dim=1, keepdim=True) + self.b
            else:
                output = self.dendrites[0].output()
                for synapse in self.dendrites[1:]:
                    output = output + synapse.output()
                output = output + self.b
        if self.activation is not None:
            output = self.activation(output)
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
        # assert len(value.shape) == 1
        self._output = value

    def _compute_output(self) -> torch.Tensor:
        raise RuntimeError("Should never be called")


class RegularNeuron(DestinationNeuron, SourceNeuron):
    def __init__(self, context: "Context", activation):
        super().__init__(context, activation)


class Synapse(GraphNode):
    def __init__(self, source: SourceNeuron, destination: Union[DestinationNeuron, RegularNeuron]):
        assert source != destination
        self.source = source
        self.destination = destination
        self.context = source.context
        assert source not in (synapse.source for synapse in destination.dendrites), "Connection already exists"
        destination.dendrites.append(self)
        assert destination not in (synapse.destination for synapse in source.axons), "Connection already exists"
        source.axons.append(self)
        self.k = source.context.obtain_float_parameter("s")
        self.optimizer = optimizer.SGD1(self.k)

    def init_weight(self):
        v = math.sqrt(1 / len(self.destination.dendrites))
        with torch.no_grad():
            self.k[...] = self.context.random.uniform(-v, v)

    def on_grad_update(self):
        self.optimizer.step()

    def output(self):
        output = self.k * self.source.compute_output()
        return output

    def get_adjacent_nodes(self) -> List[GraphNode]:
        return [self.source]


class Context:
    def __init__(self, module: nn.Module, seed):
        self.random = random.Random(seed)
        self.module = module
        self.n_params = 0
        self.name_counters = {}
        self.decay = 0.0
        self.reduce_sum_computation = False

    def obtain_float_parameter(self, name_prefix: str) -> nn.Parameter:
        if name_prefix not in self.name_counters:
            self.name_counters[name_prefix] = -1
        self.name_counters[name_prefix] += 1
        name = f"{name_prefix}{self.name_counters[name_prefix]}"
        self.n_params += 1
        param = nn.Parameter(torch.tensor(0.0))
        self.module.register_parameter(name, param)
        return param


class LiveNet(nn.Module):
    def __init__(self, n_inputs, n_middle, n_outputs, seed=0):
        super().__init__()
        self.context = Context(self, seed)
        self.inputs = [DataNeuron(self.context) for _ in range(n_inputs)]
        self.outputs = [DestinationNeuron(self.context, activation=None) for _ in range(n_outputs)]
        self.root = NodesHolder(self.outputs)
        if n_middle is None:
            for input_ in self.inputs:
                for output in self.outputs:
                    input_.connect_to(output)
        else:
            for i in range(n_middle):
                neuron = RegularNeuron(self.context, activation=torch.nn.ReLU())
                for input_ in self.inputs:
                    input_.connect_to(neuron)
                for output in self.outputs:
                    output.connect_from(neuron)
        self.root.visit("init_weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2, "Invalid input shape"
        self.root.visit("clear_output")
        for i in range(x.shape[1]):
            self.inputs[i].set_output(x[:, i: i + 1])
        outputs = [o.compute_output() for o in self.outputs]
        y = torch.cat(outputs, dim=1)
        return y

    def internal_loss(self):
        return torch.tensor(0.0)

    def visit(self, func):
        self.root.visit(func)

    def zero_grad(self, set_to_none: bool = True):
        assert set_to_none is False
        self.root.visit("optimizer.zero_grad")

    def on_grad_update(self):
        self.root.visit("on_grad_update")

    def input_shape(self):
        return torch.Size([len(self.inputs)])


class LiveNetOptimizer:
    def __init__(self, network: LiveNet):
        self.network = network

    def zero_grad(self):
        self.network.zero_grad(False)

    def step(self):
        self.network.on_grad_update()


def export_onnx(model: nn.Module):
    dummy_input = torch.zeros((1, *net.input_shape()))
    torch.onnx.export(model, dummy_input, "/home/spometun/model.onnx", verbose=False)


if __name__ == "__main__":
    utils.set_seed()
    x = torch.tensor([[0., 0], [0, 1], [1, 0], [1, 1]])
    net = LiveNet(2, 4, 1)
    net.context.reduce_sum_computation = True
    net.forward(x)
    net.forward(x)

    LOG('a')
    export_onnx(net)
    LOG('b')

    p = [p for p in net.named_parameters()]
    s1 = net.get_parameter("s1")

    LOG("Here")
    # net = LiveNet(2, 1)
