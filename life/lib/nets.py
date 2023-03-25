import math

import torch
from torch import nn as nn

from life import lib as lib


def criterion(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    y_hat = torch.nn.functional.sigmoid(logits)
    return torch.nn.functional.binary_cross_entropy(y_hat, label) / math.log(2)


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
    def __init__(self):
        super(ODD, self).__init__()
        self.linear1 = nn.Linear(1, 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2, 1)
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


def create_livenet_odd():
    network = lib.livenet.LiveNet(1, 2, 1)
    with torch.no_grad():
        network.inputs[0].axons[0].k[...] = torch.tensor(10)
        network.inputs[0].axons[1].k[...] = torch.tensor(-10)
        network.inputs[0].axons[0].destination.b[...] = torch.tensor(-15)
        network.inputs[0].axons[1].destination.b[...] = torch.tensor(2)
        network.inputs[0].axons[0].destination.axons[0].k[...] = torch.tensor(-1)
        network.inputs[0].axons[1].destination.axons[0].k[...] = torch.tensor(-1)
        network.inputs[0].axons[1].destination.axons[0].destination.b[...] = torch.tensor(1)
    return network
