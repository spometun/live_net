import math

import torch
from torch import nn as nn
from life.lib.simple_log import LOG

from life import lib as lib


def criterion_1(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    y_hat = torch.nn.functional.sigmoid(logits)
    assert y_hat.detach().shape[1] == 1
    label = label.float()
    return torch.nn.functional.binary_cross_entropy(y_hat, label) / math.log(2)


def criterion_n(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = torch.squeeze(labels, 1)
    return nn.functional.cross_entropy(inputs, labels) / math.log(2)


class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.linear = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2, 1)

    def forward(self, input_):
        x = self.linear(input_)
        activation = self.relu(x)
        yh = self.linear2(activation)
        return yh


class ODD(nn.Module):
    def __init__(self, l1=0.0):
        super(ODD, self).__init__()
        self.linear1 = nn.Linear(1, 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2, 1)
        self.alpha_l1 = l1
        with torch.no_grad():
            p = [p for p in self.linear1.parameters()]
            p[0][...] = torch.tensor([[10], [-10]])
            p[1][...] = torch.tensor([-15, 2])
            p = [p for p in self.linear2.parameters()]
            p[0][...] = torch.tensor([[-1, -1]])
            p[1][...] = torch.tensor([1])

    def forward(self, input_):
        x = self.linear1(input_)
        a = self.relu(x)
        x = self.linear2(a)
        y_hat = x
        return y_hat

    def internal_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.)
        for param in self.parameters():
            if len(param.shape) > 1:
                loss += self.alpha_l1 * torch.sum(torch.abs(param))
        return loss


class PYRAMID(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2, 1)
        with torch.no_grad():
            p = [p for p in self.linear1.parameters()]
            p[0][...] = torch.tensor([[-10, -11, -12], [10, 11, 12]])
            p[1][...] = torch.tensor([5, -25])
            p = [p for p in self.linear2.parameters()]
            p[0][...] = torch.tensor([[1, 1]])
            p[1][...] = torch.tensor([0])

    def forward(self, input_):
        x = self.linear1(input_)
        a = self.relu(x)
        x = self.linear2(a)
        y_hat = x
        return y_hat


class PERCEPTRON(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, l1=0.0):
        super().__init__()
        self.alpha_l1 = l1
        self.linear1 = nn.Linear(n_inputs, n_outputs)

    def forward(self, input_):
        x = self.linear1(input_)
        y_hat = x
        return y_hat

    def input_shape(self):
        return torch.Size([self.linear1.in_features])

    def internal_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.)
        for param in self.parameters():
            if len(param.shape) > 1:
                loss += self.alpha_l1 * torch.sum(torch.abs(param))
        return loss


def create_livenet_odd(l1=0.0):
    network = lib.livenet.LiveNet(1, 2, 1)
    network.context.alpha_l1 = l1
    with torch.no_grad():
        network.inputs[0].axons[0].k[...] = torch.tensor(10)
        network.inputs[0].axons[1].k[...] = torch.tensor(-10)
        network.inputs[0].axons[0].destination.b[...] = torch.tensor(-15)
        network.inputs[0].axons[1].destination.b[...] = torch.tensor(2)
        network.inputs[0].axons[0].destination.axons[0].k[...] = torch.tensor(-1)
        network.inputs[0].axons[1].destination.axons[0].k[...] = torch.tensor(-1)
        network.inputs[0].axons[1].destination.axons[0].destination.b[...] = torch.tensor(1)
    return network


def create_livenet_odd_2():
    network = lib.livenet.LiveNet(1, 2, 2)
    with torch.no_grad():
        network.inputs[0].axons[0].k[...] = torch.tensor(2.2)
        network.inputs[0].axons[1].k[...] = torch.tensor(-2.1)

        network.inputs[0].axons[0].destination.b[...] = torch.tensor(-3.1)
        network.inputs[0].axons[1].destination.b[...] = torch.tensor(1)

        network.inputs[0].axons[0].destination.axons[0].k[...] = torch.tensor(1.1)
        network.inputs[0].axons[0].destination.axons[1].k[...] = torch.tensor(-1.2)
        network.inputs[0].axons[1].destination.axons[0].k[...] = torch.tensor(1.3)
        network.inputs[0].axons[1].destination.axons[1].k[...] = torch.tensor(-1)

    return network


def create_livenet_pyramid():
    network = lib.livenet.LiveNet(3, 2, 1)
    with torch.no_grad():
        network.inputs[0].axons[0].k[...] = torch.tensor(-10)
        network.inputs[0].axons[1].k[...] = torch.tensor(10)
        network.inputs[1].axons[0].k[...] = torch.tensor(-11)
        network.inputs[1].axons[1].k[...] = torch.tensor(11)
        network.inputs[2].axons[0].k[...] = torch.tensor(-12)
        network.inputs[2].axons[1].k[...] = torch.tensor(12)

        network.inputs[0].axons[0].destination.b[...] = torch.tensor(5)
        network.inputs[0].axons[1].destination.b[...] = torch.tensor(-25)
        network.inputs[0].axons[0].destination.axons[0].k[...] = torch.tensor(1)
        network.inputs[0].axons[1].destination.axons[0].k[...] = torch.tensor(1)
        network.inputs[0].axons[1].destination.axons[0].destination.b[...] = torch.tensor(0)
    return network
