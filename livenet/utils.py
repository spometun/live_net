import torch


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
