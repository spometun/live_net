from typing import Optional
import torch
from life.lib.simple_log import LOG
from life.lib.livenet import LiveNet
import life.lib.livenet


class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters, lr=0.01):
        self._params: list[torch.nn.parameter.Parameter] = list(parameters)
        self._lr = lr
        # super(MyOptimizer, self).__init__(parameters, {})

    def zero_grad(self) -> None:
        for p in self._params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self._params:
            # p.data = p.data * (1 - torch.sign(p.grad.data) * self._lr)
            p.data.add_(p.grad.data, alpha=-self._lr)


class SGD1:
    def __init__(self, parameter: torch.Tensor, context: life.lib.livenet.Context):
        self.parameter = parameter
        self.context = context

    def zero_grad(self):
        with torch.no_grad():
            self.parameter.grad.zero_()

    def step(self):
        lr = self.context.learning_rate
        with torch.no_grad():
            self.parameter += -lr * self.parameter.grad


def optimizer_with_lr_property(opt_class: torch.optim.Optimizer, *args, **kwargs):
    class _OptimizerWithProperty(opt_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @property
        def learning_rate(self):
            assert len(self.param_groups) == 1
            return self.param_groups[0]["lr"]

        @learning_rate.setter
        def learning_rate(self, value):
            assert len(self.param_groups) == 1
            self.param_groups[0]["lr"] = value

    return _OptimizerWithProperty(*args, **kwargs)


class LiveNetOptimizer:
    def __init__(self, network: LiveNet):
        self.network = network

    def zero_grad(self):
        self.network.zero_grad(False)

    def step(self):
        self.network.on_grad_update()

    @property
    def learning_rate(self):
        return self.network.context.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.network.context.learning_rate = value

