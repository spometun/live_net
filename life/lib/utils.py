from typing import Callable
import random
import numpy as np
import torch
from torch import Tensor
import time
import math


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ValueHolder:
    def __init__(self, value=None):
        self.value = value


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


def get_parameters_copy(network: torch.nn.Module):
    params = [p.detach().clone() for p in network.parameters()]
    return params


def get_parameters_dict(network: torch.nn.Module, clone=True):
    res = {}
    for name, param in network.named_parameters():
        param = param.detach()
        if clone:
            param = param.clone()
        res[name] = param.numpy()
    return res


def get_gradients_dict(network: torch.nn.Module):
    res = {}
    for name, param in network.named_parameters():
        with torch.no_grad():
            res[name] = param.grad
    return res


def set_parameters(network: torch.nn.Module, params: list):
    net_params = [p for p in network.parameters()]
    assert len(net_params) == len(params)
    with torch.no_grad():
        for i, p in enumerate(net_params):
            net_params[i][...] = torch.tensor(params[i][...])


def add_noise_to_params(params: list, a, b):
    with torch.no_grad():
        for i in range(len(params)):
            p = params[i]
            try:
                p = p.numpy()
            except AttributeError:
                pass

            with np.nditer(p, flags=['multi_index']) as it:
                for _ in it:
                    index = it.multi_index
                    val = float(p[*index])
                    val = random.uniform(-b, b) + (1 + random.uniform(-a, a)) * val
                    params[i][*index] = val


