from typing import Optional
import torch


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
    def __init__(self, parameter: torch.Tensor, learning_rate=0.01):
        self.parameter = parameter
        self.learning_rate = learning_rate

    def zero_grad(self):
        with torch.no_grad():
            self.parameter.grad.zero_()

    def step(self):
        with torch.no_grad():
            self.parameter += -self.learning_rate * self.parameter.grad
            pass

