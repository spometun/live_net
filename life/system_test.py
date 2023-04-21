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
    plt.ion()
    lib.utils.set_seed()
    print(torch.__version__)
    np.set_printoptions(precision=3)
    lib.utils.set_seed()
    downscale = 28
    train_x, train_y = lib.datasets.to_plain_odd(*lib.datasets.get_mnist_test(), downscale=downscale)
    network = lib.livenet.LiveNet(784 // (downscale * downscale), 1, 2)

    batch_iterator = lib.gen_utils.batch_iterator(train_x, train_y, batch_size=100)
    criterion = lib.nets.criterion_n
    optimizer = lib.nets.create_optimizer(network)
    optimizer.learning_rate = 0.001
    trainer = lib.trainer.Trainer(network, batch_iterator, criterion, optimizer, epoch_size=100)

    network.context.alpha_l1 = 0.9

    # network.inputs[0].axons[0].die()
    # network.outputs[1].dendrites[0].die()
    # network.outputs[0].dendrites[0].die()
    #
    optimizer.zero_grad()
    optimizer.step()
    # network.on_grad_update()
    # trainer.step(500)

    # for src in network.inputs:
    #     for s in src.axons:
    #         s.die()
    # for dest in network.outputs:
    #     for s in dest.dendrites:
    #         s.die()


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


