import random
import numpy as np
import torch


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ValueHolder:
    def __init__(self, value=None):
        self.value = value


def get_parameters_copy(network: torch.nn.Module):
    params = [p.detach().clone() for p in network.parameters()]
    return params


def set_parameters(network: torch.nn.Module, params: list):
    net_params = [p for p in network.parameters()]
    assert len(net_params) == len(params)
    with torch.no_grad():
        for i, p in enumerate(net_params):
            net_params[i][...] = torch.tensor(params[i][...])

'''
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
'''


def broadcast_dimensions(tensors: list[torch.Tensor], target_shape: tuple = None):
    if target_shape is None:
        for tensor in tensors:
            if len(tensor.shape) > 0:
                target_shape = tensor.shape
                break
    if target_shape is None:
        return

    for i, tensor in enumerate(tensors):
        if tensor.shape == target_shape:
            continue
        assert len(tensor.shape) == 0
        tensors[i] = tensor.repeat(*target_shape)
