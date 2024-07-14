from typing import Optional
import torch
from ai_libs.simple_log import LOG
import math

from livenet.core.observability import LifeStatContributor


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


class SGDForParameter:
    def __init__(self, parameter: torch.Tensor, context):
        self.parameter = parameter
        self.context = context

    def zero_grad(self):
        with torch.no_grad():
            self.parameter.grad.zero_()

    def step(self):
        lr = self.context.learning_rate
        with torch.no_grad():
            self.parameter += -lr * self.parameter.grad


class AdamForParameter(LifeStatContributor):
    def __init__(self, parameter: torch.Tensor, context,
                 betas: tuple, epsilon=1e-8):
        self.parameter = parameter
        self.name = parameter.livenet_name
        self.context = context
        self.t = 0
        self.b1t = 1.
        self.b2t = 1.
        self.epsilon = epsilon
        self.mt = torch.zeros_like(self.parameter)
        self.vt = torch.zeros_like(self.parameter)
        assert len(betas) == 2
        assert 0 <= betas[0] <= 1
        assert 0 <= betas[1] <= 1
        self.b1 = betas[0]
        self.b2 = betas[1]

    def zero_grad(self):
        with torch.no_grad():
            if self.parameter.grad is not None:
                self.parameter.grad.zero_()

    def step(self):
        if self.parameter.requires_grad:
            with torch.no_grad():
                lr = self.context.learning_rate
                self.t += 1
                self.b1t *= self.b1
                self.b2t *= self.b2
                g = self.parameter.grad
                assert math.isfinite(g)
                self.add_life_stat_entry("gradient", g)
                self.mt = self.b1 * self.mt + (1 - self.b1) * g
                self.vt = self.b2 * self.vt + (1 - self.b2) * (g * g)
                mt = self.mt / (1 - self.b1t)
                vt = self.vt / (1 - self.b2t)
                delta = -lr * mt / (torch.sqrt(vt) + self.epsilon)
                self.add_life_stat_entry("delta", delta)
                self.parameter += delta
                assert math.isfinite(self.parameter.item())
        self.add_life_stat_entry("parameter", self.parameter)


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
    def __init__(self, network: "LiveNet", lr):
        self.network = network
        self.network.context.learning_rate = lr

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

