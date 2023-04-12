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
from life.lib.utils import ValueHolder


from life.lib.simple_log import LOG
import life.lib.optimizer as optimizer


def test_dev():
    print("O merciful God")


class Neuron(GraphNode):
    def __init__(self, context: "Context"):
        super().__init__()
        self.context = context
        self.id = context.get_id()
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
    def get_adjacent_nodes(self) -> List["GraphNode"]:
        return []


class DestinationNeuron(Neuron):
    def __init__(self, context: "Context", activation):
        super().__init__(context)
        self.dendrites: List[Synapse] = []
        self.b = context.obtain_float_parameter(f"{self.id}")
        self.optimizer = self.context.optimizer_class(self.b, context, **self.context.optimizer_init_kwargs)
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
    def get_adjacent_nodes(self) -> List["GraphNode"]:
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


class LivenessObserver:
    def __init__(self):
        self.dead = False
        self.threshold = 0.05;
        self.weight = 0.05
        self.value = math.pow(1 - self.weight, -100)

    def put(self, x: float):
        self.value = (1 - self.weight) * self.value + self.weight * x

    def looks_ok(self):
        return self.value >= 0.05


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
        self.name = f"{source.id}->{destination.id}"
        self.k = self.context.obtain_float_parameter(self.name)
        self.random_constant = self.context.random.uniform(-1, 1)
        self.optimizer = self.context.optimizer_class(self.k, self.context, **self.context.optimizer_init_kwargs)
        self.liveness_observer = LivenessObserver()

    def init_weight(self):
        v = math.sqrt(1 / len(self.destination.dendrites))
        with torch.no_grad():
            self.k[...] = self.context.random.uniform(-v, v)

    def on_grad_update(self):
        self.optimizer.step()
        self.liveness_observer.put(self.k.item())
        if not self.liveness_observer.looks_ok():
            if not self.liveness_observer.dead:
                LOG(f"{self.name} died")
            self.liveness_observer.dead = True

    def output(self):
        output = self.k * self.source.compute_output()
        return output

    def internal_loss(self, loss: ValueHolder):
        alpha_l1 = self.context.alpha_l1 * (1 + 0.1 * self.random_constant)
        loss.value += alpha_l1 * torch.abs(self.k)

    def get_adjacent_nodes(self) -> List[GraphNode]:
        return [self.source]


class Context:
    def __init__(self, module: nn.Module, seed):
        self.random = random.Random(seed)
        self.module = module
        self.n_params = 0
        self.learning_rate = None
        self.optimizer_class = optimizer.AdamLiveNet
        self.optimizer_init_kwargs = {"betas": (0.0, 0.95)}
        self.alpha_l1 = 0.0
        self.id_counter = 0
        self.reduce_sum_computation = False

    def get_id(self):
        id_ = self.id_counter
        self.id_counter += 1
        return id_

    def obtain_float_parameter(self, name: str) -> nn.Parameter:
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
        assert len(self.inputs) == x.shape[1]
        self.root.visit("clear_output")
        for i in range(x.shape[1]):
            self.inputs[i].set_output(x[:, i: i + 1])
        outputs = [o.compute_output() for o in self.outputs]
        y = torch.cat(outputs, dim=1)
        return y

    def internal_loss(self):
        loss = ValueHolder(torch.tensor(0.0))
        self.root.visit("internal_loss", loss)
        return loss.value

    def visit(self, func):
        self.root.visit(func)

    def zero_grad(self, set_to_none: bool = True):
        assert set_to_none is False
        self.root.visit("optimizer.zero_grad")

    def on_grad_update(self):
        self.root.visit("on_grad_update")

    def input_shape(self):
        return torch.Size([len(self.inputs)])


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
