import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import importlib
import life.lib
import life.lib as lib
import typing
importlib.reload(lib)
LOG = lib.simple_log.LOG
import math
from matplotlib import pyplot as plt
import pytest


def test_die():
    plt.ion()
    lib.utils.set_seed()
    print(torch.__version__)
    np.set_printoptions(precision=3)
    lib.utils.set_seed()
    downscale = 14
    train_x, train_y = lib.datasets.to_plain_odd(*lib.datasets.get_mnist_test(), downscale=downscale)
    network = lib.livenet.LiveNet(784 // (downscale * downscale), 7, 2)

    batch_iterator = lib.gen_utils.batch_iterator(train_x, train_y, batch_size=100)
    criterion = lib.nets.criterion_n
    optimizer = lib.nets.create_optimizer(network)
    optimizer.learning_rate = 0.001
    trainer = lib.trainer.Trainer(network, batch_iterator, criterion, optimizer, epoch_size=100)

    network.context.alpha_l1 = 0.1
    trainer.step(500)

