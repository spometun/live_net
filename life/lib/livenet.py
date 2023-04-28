import typing
from typing import List, Union
# from overrides import override
import abc
import torch
import torch.nn as nn
import math
import random

from life.lib.death import LivenessObserver, DeathStat
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
        self.name = context.get_name(type(self))
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

    def die(self):
        LOG(f"There is no death for {self.name}")


# noinspection PyAbstractClass
class SourceNeuron(Neuron):
    def __init__(self, context: "Context"):
        super().__init__(context)
        self.axons: List[Synapse] = []

    def add_axon(self, synapse: "Synapse"):
        assert synapse not in self.axons, "Internal error"
        self.axons.append(synapse)

    def remove_axon(self, synapse: "Synapse"):
        assert synapse in self.axons, "Internal error"
        self.axons.remove(synapse)
        if len(self.axons) == 0:
            LOG(f"{self.name} became useless at tick {self.context.tick}")
            self.die()

    def connect_to(self, destination: "DestinationNeuron"):  # high-level helper function
        synapse = Synapse(self, destination)
        return synapse

    # @override
    def get_adjacent_nodes(self) -> List["GraphNode"]:
        return []


class DestinationNeuron(Neuron):
    def __init__(self, context: "Context", activation):
        super().__init__(context)
        self.dendrites: List[Synapse] = []
        self.b = context.obtain_float_parameter(self.name)
        self.optimizer = self.context.optimizer_class(self.b, context, **self.context.optimizer_init_kwargs)
        self.activation = activation

    def on_grad_update(self):
        self.optimizer.step()

    def _compute_output(self) -> torch.Tensor:
        outputs = [self.b]
        if self.context.reduce_sum_computation:
            for synapse in self.dendrites:
                outputs.append(synapse.output())
            utils.broadcast_dimensions(outputs)
            ndim = len(outputs[0].shape)
            if ndim == 0:
                outputs = [o.reshape(1, 1) for o in outputs]
            assert len(outputs[0].shape) == 2, "Internal error"
            all_ = torch.cat(outputs, dim=1)
            output = torch.sum(all_, dim=1, keepdim=True)
            if ndim == 0:
                output = output.reshape([])
        else:
            output = self.b
            for synapse in self.dendrites:
                output = output + synapse.output()
        if self.activation is not None:
            output = self.activation(output)
        return output

    def add_dendrite(self, synapse: "Synapse"):
        assert synapse not in self.dendrites, "Internal error"
        self.dendrites.append(synapse)

    def remove_dendrite(self, synapse: "Synapse"):
        assert synapse in self.dendrites, "Internal error"
        self.dendrites.remove(synapse)
        if len(self.dendrites) == 0:
            self.context.death_stat.on_dangle_neuron(self)

    def connect_from(self, source: SourceNeuron):  # high-level helper function
        synapse = Synapse(source, self)
        return synapse

    # @override
    def die(self):
        LOG(f"killing {self.name} with b={self.b.item():.3f}, tick={self.context.tick}")
        while len(self.dendrites) > 0:
            self.dendrites[0].die()
        self.context.death_stat.off_dangle_neuron(self)
        self.context.remove_parameter(self.name)

    # @override
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

    # @override
    def die(self):
        assert len(self.axons) == 0, "Internal error: Wouldn't kill neuron with at least one axon alive"
        assert issubclass(RegularNeuron, DestinationNeuron)
        super(RegularNeuron, self).die()


class Synapse(GraphNode):
    def __init__(self, source: SourceNeuron, destination: Union[DestinationNeuron, RegularNeuron]):
        assert source != destination
        assert source not in (synapse.source for synapse in destination.dendrites), "Connection already exists"
        assert destination not in (synapse.destination for synapse in source.axons), "Connection already exists"
        self.source = source
        self.destination = destination
        context = source.context
        self.context = context
        self.name = f"{source.name}->{destination.name}"
        self.k = context.obtain_float_parameter(self.name)
        self.random_constant = context.random.uniform(-1, 1)
        self.optimizer = context.optimizer_class(self.k, context, **context.optimizer_init_kwargs)
        self.liveness_observer = LivenessObserver(context, self.k)
        source.add_axon(self)
        destination.add_dendrite(self)

    def init_weight(self):
        assert self.source is not None and self.destination is not None, "Internal error"
        v = math.sqrt(1 / len(self.destination.dendrites))
        with torch.no_grad():
            self.k[...] = self.context.random.uniform(-v, v)

    def on_grad_update(self):
        assert self.source is not None and self.destination is not None, "Internal error"
        self.optimizer.step()
        self.liveness_observer.put(self.k.item())
        status = self.liveness_observer.status()
        if status == -1:
            self.die()
        if status == 1:
            LOG(f"{self.name} didn't die because of not small values in it's history")

    def die(self):
        assert self.source is not None and self.destination is not None, "Internal error"
        LOG(f"killing {self.name} at tick {self.context.tick} with k={self.k.item():.3f}")
        dst = self.destination
        self.destination = None
        src = self.source
        self.source = None
        dst.remove_dendrite(self)
        src.remove_axon(self)
        self.context.remove_parameter(self.name)

    def output(self):
        assert self.source is not None and self.destination is not None, "Internal error"
        output = self.k * self.source.compute_output()
        return output

    def internal_loss(self, loss: ValueHolder):
        assert self.source is not None and self.destination is not None, "Internal error"
        alpha_l1 = self.context.alpha_l1 * (1 + 0.1 * self.random_constant)
        loss.value += alpha_l1 * torch.abs(self.k)

    def get_adjacent_nodes(self) -> List[GraphNode]:
        if self.source is not None:
            return [self.source]
        else:
            # dead synapse will be inquired for children during the visit it died
            # because of Graph visit logic
            return []


class Context:
    def __init__(self, module: nn.Module, seed):
        self.random = random.Random(seed)
        self.module = module
        self.n_params = 0
        self.learning_rate = None
        self.optimizer_class = optimizer.AdamLiveNet
        self.optimizer_init_kwargs = {"betas": (0.0, 0.95)}
        self.alpha_l1 = 0.0
        self.name_counters = {"S": 0, "D": 0, "N": 0}
        self.death_stat = DeathStat()
        self.tick = 0
        self.reduce_sum_computation = False

    def get_name(self, cls):
        match cls.__name__:
            case "RegularNeuron":
               key = "N"
            case "DataNeuron":
                key = "S"
            case "DestinationNeuron":
                key = "D"
            case _:
                assert False, "Internal error"
        res = f"{key}{self.name_counters[key]}"
        self.name_counters[key] += 1
        return res

    def obtain_float_parameter(self, name: str) -> nn.Parameter:
        param = nn.Parameter(torch.tensor(0.0))
        self.module.register_parameter(name, param)
        return param

    def remove_parameter(self, name: str):
        self.module._parameters.pop(name)


class LiveNet(nn.Module):
    def __init__(self, n_inputs, n_middle, n_outputs, seed=0):
        super().__init__()
        self.context = Context(self, seed)
        self.inputs = [DataNeuron(self.context) for _ in range(n_inputs)]
        self.outputs = [DestinationNeuron(self.context, activation=None) for _ in range(n_outputs)]
        self.root = NodesHolder("root", self.outputs)
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
        utils.broadcast_dimensions(outputs, (x.shape[0], 1))
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
        self.context.tick += 1

    def input_shape(self):
        return torch.Size([len(self.inputs)])


def export_onnx(model: nn.Module, path):
    dummy_input = torch.zeros((1, *model.input_shape()))
    torch.onnx.export(model, dummy_input, path, verbose=False)


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


