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
