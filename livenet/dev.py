# PLAN:
# 1. Create simple odd network, see how it trains, which neurons deid,
# activations of those who remain, death sign stats, etc.
# 2. Create downscaled CIFAR network, see how it trains, death, stats, etc.
# 3. Refactor to have core + high-level structure
# may be do 3. first, at least to 3. before improving/debugging deaths on 1. and 2.
# 4. Solve Context design and/or death logic design (what is provided into constructors) if needed
# 5. Provide OBSERVABILITY and after that/alongside with that conduct meaningful experiments

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import importlib
from . import core
import typing
importlib.reload(core)
from simple_log import LOG
import math
from matplotlib import pyplot as plt
import pytest


def test_dev():
    # simple_log.level = simple_log.LogLevel.DEBUG
    train_x, train_y = core.datasets.get_odd()
    network = core.nets.create_livenet_odd_2()
    res = network(train_x)
    network.context.regularization_l1 = 0.05
    batch_iterator = core.gen_utils.batch_iterator(train_x, train_y, batch_size=len(train_x))
    criterion = core.nets.criterion_n
    optimizer = core.optimizers.LiveNetOptimizer(network, lr=0.02)
    trainer = core.net_trainer.NetTrainer(network, batch_iterator, criterion, optimizer, epoch_size=50)
    trainer.step(1001)
    scores = nn.functional.softmax(network(train_x), dim=1).detach().numpy()
    LOG(scores)
    prediction = np.argmax(scores, axis=1)
    assert np.all(prediction == train_y.numpy().squeeze(1))
    core.livenet.export_onnx(network, "/home/spometun/table/home/net.onnx")

