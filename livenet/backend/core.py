import typing
from typing import Any
from abc import ABC
from typing import List, override
import abc

import numpy as np
import torch
import torch.nn as nn
import math
import random

from .death import LivenessObserver
from livenet.backend.observability import TopologyStat, LifeStatContributor
from .graph import GraphNode, NodesHolder
from .utils import ValueHolder

from ai_libs.simple_log import LOG, LOGD
from . import optimizers
from . import utils


class NeuralBase(GraphNode, LifeStatContributor):
    def __init__(self, context: "Context", only_one_output_request=False):
        assert context is not None
        assert self.name is not None, "Internal error"
        super().__init__()
        self._output = None
        self.context = context
        self.only_one_output_request = only_one_output_request

    @typing.final
    def compute_output(self) -> torch.Tensor:
        # output is (batch, 1) tensor of floats,
        # or float scalar in rare cases (e.g. in case of dangle neuron - the one with zero dendrites)
        if self.only_one_output_request:
            assert self._output is None, f"Internal error. Called compute_output on {self.__class__.__name__} more than once"
        if self._output is None:
            self._output = self._compute_output()
            with torch.no_grad():
                max_output = np.max(np.abs(self._output.detach().numpy()))
                self.add_life_stat_entry("output_max", max_output)
        return self._output

    @typing.final
    def clear_output(self):
        self._output = None

    @abc.abstractmethod
    def _compute_output(self) -> torch.Tensor: ...

    def zero_grad(self): ...

    def on_grad_update(self): ...

    def init_weight(self): ...

    def die(self):
        LOG(f"{self.name} die BaseNeural")
        self.context.remove_parameter(self.name)


class SourceNeuron(NeuralBase, ABC):
    def __init__(self, context: "Context"):
        LOGD(f"Source neuron init {self.name}")
        super().__init__(context)
        self.axons: List[Synapse] = []
        self.context.topology_stat.on_useless_neuron(self)

    def add_axon(self, synapse: "Synapse"):
        assert synapse not in self.axons, "Internal error"
        if len(self.axons) == 0:
            self.context.topology_stat.off_useless_neuron(self)
        self.axons.append(synapse)

    def remove_axon(self, synapse: "Synapse"):
        assert synapse in self.axons, "Internal error"
        self.axons.remove(synapse)
        if len(self.axons) == 0:
            LOG(f"{self.name} became useless and will die at tick {self.context.tick}")
            self.context.topology_stat.on_useless_neuron(self)
            self.die()

    def connect_to(self, destination: "DestinationNeuron"):  # high-level helper function
        synapse = Synapse(self, destination)
        return synapse

    @override
    def die(self):
        LOG(f"killing SourceNeuron {self.name} tick={self.context.tick}")
        assert len(self.axons) == 0, "Internal error: Wouldn't kill neuron with at least one axon alive"
        self.context.topology_stat.off_useless_neuron(self)
        super().die()


class DestinationNeuron(NeuralBase):
    def __init__(self, context: "Context", activation):
        try:
            self.name
        except AttributeError:
            self.name = context.get_name("D")
        super().__init__(context)
        LOGD(f"Destination neuron init {self.name}")
        self.dendrites: List[Synapse] = []
        self.b = context.obtain_float_parameter(self.name)
        self.optimizer = self.context.optimizer_class(self.b, context, **self.context.optimizer_init_kwargs)
        if activation is None:
            activation = lambda x: x
        self.activation = activation
        self.context.topology_stat.on_dangle_neuron(self)

    @override
    def zero_grad(self):
        self.optimizer.zero_grad()

    @override
    def on_grad_update(self):
        self.optimizer.step()

    @override
    def _compute_output(self) -> torch.Tensor:
        if self.context.reduce_sum_computation:
            outputs = [self.b]
            for synapse in self.dendrites:
                outputs.append(synapse.compute_output())
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
                output = output + synapse.compute_output()
        active_output = self.activation(output)
        with torch.no_grad():
            ndim = len(output.shape)
            if ndim == 0:
                n = 1
            else:
                assert ndim == 2, "Internal error"
                n = len(output)
            ratio_low_cut = torch.sum(output < active_output).item() / n
            ratio_high_cut = torch.sum(output > active_output).item() / n
            self.add_life_stat_entry("output_low_cut_ratio", ratio_low_cut)
            self.add_life_stat_entry("output_high_cut_ratio", ratio_high_cut)

        return active_output

    def add_dendrite(self, synapse: "Synapse"):
        assert synapse not in self.dendrites, "Internal error"
        if len(self.dendrites) == 0:
            self.context.topology_stat.off_dangle_neuron(self)
        self.dendrites.append(synapse)

    def remove_dendrite(self, synapse: "Synapse"):
        assert synapse in self.dendrites, "Internal error"
        self.dendrites.remove(synapse)
        if len(self.dendrites) == 0:
            self.context.topology_stat.on_dangle_neuron(self)
            # TODO: Want to kill self safely (if it is possible, consider b)?

    def connect_from(self, source: SourceNeuron):  # high-level helper function
        synapse = Synapse(source, self)
        return synapse

    @override
    def get_adjacent_nodes(self) -> List[GraphNode]:
        return self.dendrites

    @override
    def die(self):
        LOG(f"killing DestinationNeuron {self.name} with b={self.b.item():.3f}, tick={self.context.tick}")
        # TODO: 'b' must be added to destination's b, or be close to zero, or handled somehow in other way
        while len(self.dendrites) > 0:
            self.dendrites[-1].die()
        self.context.topology_stat.off_dangle_neuron(self)
        super().die()


class InputNeuron(SourceNeuron):
    def __init__(self, context: "Context"):
        try:
            self.name
        except AttributeError:
            self.name = context.get_name("I")
        super().__init__(context)
        LOGD(f"InputNeuron init {self.name}")

    def set_output(self, value: torch.Tensor):
        assert value is not None, "Invalid input"
        # assert len(value.shape) == 1
        self._output = value

    @override
    def _compute_output(self) -> torch.Tensor:
        raise RuntimeError("Should never be called")

    @override
    def get_adjacent_nodes(self) -> List[GraphNode]:
        return []

    @override
    def die(self):
        LOG(f"InputNeuron {self.name} death is no-op")


# todo: make all __init__ args kwargs for more flexible inheritance options
class RegularNeuron(DestinationNeuron, SourceNeuron):
    def __init__(self, context: "Context", activation):
        # todo: use try/except when add any subclass of RegularNeuron
        self.name = context.get_name("N")
        super().__init__(context=context, activation=activation)


class Synapse(NeuralBase):
    def __init__(self, source: SourceNeuron, destination: DestinationNeuron):
        self.name = f"{source.name}->{destination.name}"
        context = source.context
        super().__init__(context=context, only_one_output_request=True)
        assert source != destination
        assert source not in (synapse.source for synapse in destination.dendrites), "Connection already exists"
        assert destination not in (synapse.destination for synapse in source.axons), "Connection already exists"
        self.source = source
        self.destination = destination
        assert source.context == destination.context
        self.context = context
        self.k = context.obtain_float_parameter(self.name)
        self.random_constant = context.random.uniform(-1, 1)
        self.optimizer = context.optimizer_class(self.k, context, **context.optimizer_init_kwargs)
        self.liveness_observer = LivenessObserver(context)
        source.add_axon(self)
        destination.add_dendrite(self)

    def init_weight(self):
        assert self.source is not None and self.destination is not None, "Internal error"
        v = math.sqrt(1 / len(self.destination.dendrites))
        with torch.no_grad():
            self.k[...] = self.context.random.uniform(-v, v)

    @override
    def zero_grad(self):
        self.optimizer.zero_grad()

    @override
    def on_grad_update(self):
        assert self.source is not None and self.destination is not None, "Internal error"
        self.optimizer.step()
        self.liveness_observer.put(self.k.item())

    def kill_if_needed(self):
        status = self.liveness_observer.status()
        if status == -1:
            self.die()
        if status == 1:
            LOGD(f"{self.name} didn't die because of not small values in it's history")

    def die(self):
        assert self.source is not None and self.destination is not None, "Internal error"
        LOGD(f"killing {self.name} at tick {self.context.tick} with k={self.k.item():.3f}")
        dst = self.destination
        self.destination = None
        src = self.source
        self.source = None
        dst.remove_dendrite(self)
        src.remove_axon(self)
        super().die()

    @override
    def _compute_output(self):
        assert self.source is not None and self.destination is not None, "Internal error"
        output = self.k * self.source.compute_output()
        return output

    def internal_loss(self, loss: ValueHolder):
        assert self.source is not None and self.destination is not None, "Internal error"
        regularization_l1 = self.context.regularization_l1 * (1 + 0.1 * self.random_constant)
        loss.value += regularization_l1 * torch.abs(self.k)

    def get_adjacent_nodes(self) -> List[GraphNode]:
        if self.source is not None:
            return [self.source]
        else:
            # dead synapse will be inquired for children during the visit it died
            # because of Graph visit logic
            return []


class Context:
    def __init__(self, module: nn.Module, seed=0):
        self.module = module
        self.random = random.Random(seed)
        self.n_params = 0
        self.learning_rate = None
        # self.optimizer_class = optimizers.optimizers.AdamForParameter
        # self.optimizer_init_kwargs = {"betas": (0.0, 0.95)}
        self.optimizer_class = optimizers.optimizers.AdStepParameter
        self.optimizer_init_kwargs = {"window_length": 20}
        self.regularization_l1 = 0.0  # L1 regularization value
        self.name_counters = {}
        self.topology_stat = TopologyStat()
        self.life_stat: list[dict[str, Any]] = []
        self.tick: int = 0
        self.liveness_die_after_n_sign_changes = 5
        self.reduce_sum_computation = False

    def get_name(self, key: str):
        if key not in self.name_counters.keys():
            self.name_counters[key] = 0
        res = f"{key}{self.name_counters[key]}"
        self.name_counters[key] += 1
        return res

    def obtain_float_parameter(self, name: str) -> nn.Parameter:
        param = nn.Parameter(torch.tensor(0.0))
        self.module.register_parameter(name, param)
        param.livenet_name = name
        return param

    def remove_parameter(self, name: str):
        self.module._parameters.pop(name)


class LiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = Context(self)
        self.inputs: list[InputNeuron] = []
        self.outputs: list[DestinationNeuron] = []
        self.root = NodesHolder("root", self.outputs)
        self.mortal = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2, "Invalid input shape"
        assert len(self.inputs) == x.shape[1]
        self.root.visit_member("clear_output")
        for i in range(x.shape[1]):
            self.inputs[i].set_output(x[:, i: i + 1])
        outputs = [o.compute_output() for o in self.outputs]
        utils.broadcast_dimensions(outputs, (x.shape[0], 1))
        y = torch.cat(outputs, dim=1)
        return y

    def internal_loss(self):
        loss = ValueHolder(torch.tensor(0.0))
        self.root.visit_member("internal_loss", loss)
        return loss.value

    def zero_grad(self, set_to_none: bool = True):
        assert set_to_none is False  # this parameter only to match torch nn.Module zero_grad interface
        with torch.no_grad():
            self.root.visit_member("zero_grad")

    def on_grad_update(self):
        with torch.no_grad():
            self.root.visit_member("on_grad_update")
            if self.mortal:
                self.root.visit_member("kill_if_needed")
        self.context.tick += 1

    def input_shape(self):
        return torch.Size([len(self.inputs)])


if __name__ == "__main__":
    utils.set_seed()
