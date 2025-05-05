import math
import time
import random
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from ai_libs.simple_log import LOG

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parameters_dict(network: torch.nn.Module, clone=True):
    res = {}
    for name, param in network.named_parameters():
        param = param.detach()
        if clone:
            param = param.clone()
        res[name] = param.cpu().numpy()
    return res


def get_gradients_dict(network: torch.nn.Module):
    res = {}
    for name, param in network.named_parameters():
        with torch.no_grad():
            res[name] = param.grad
    return res


def calc_batch_times(network: torch.nn.Module,
                     criterion: Callable[[Tensor, Tensor], Tensor],
                     max_batch_size=4096):
    batch_size = 1
    sizes = []
    times = []
    while batch_size <= max_batch_size:
        input_ = torch.rand((batch_size, *network.input_shape()))
        labels = torch.zeros((batch_size, 1), dtype=torch.int64)
        n_tries = 20
        t0 = time.time()
        for i in range(n_tries):
            output = network.forward(input_)
            loss = criterion(output, labels)
            loss.backward()
        t1 = time.time()
        times.append((t1 - t0) / (n_tries * batch_size))
        sizes.append(batch_size)
        if batch_size == 1:
            batch_size = 2
        else:
            if math.log2(batch_size).is_integer():
                batch_size += batch_size // 2
            else:
                batch_size = batch_size // 3 * 4
    return sizes, times


def export_onnx(model: nn.Module, path):
    dummy_input = torch.zeros((1, *model.input_shape()))
    torch.onnx.export(model, dummy_input, path, verbose=False, do_constant_folding=False)
