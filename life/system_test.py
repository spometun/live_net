import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import importlib
import life.lib as lib
import typing
importlib.reload(lib)
LOG = lib.simple_log.LOG
import math
from matplotlib import pyplot as plt
import pytest


def test_bug():
    print(torch.__version__)
    np.set_printoptions(precision=3)
    lib.utils.set_seed()
    network = lib.livenet.LiveNet(1, None, 2)
    # network.zero_grad(False)
    network.on_grad_update()


def test_die():
    module = torch.nn.Module()
    context = lib.livenet.Context(module, 42)
    src = lib.livenet.RegularNeuron(context, activation=torch.nn.ReLU())
    neuron = lib.livenet.RegularNeuron(context, activation=torch.nn.ReLU)
    dst = lib.livenet.DestinationNeuron(context, activation=None)
    src.connect_to(neuron)
    neuron.connect_to(dst)
    dst.dendrites[0].die()
    assert context.death_stat.dangle_neurons == 1
    assert len(src.axons) == 0


