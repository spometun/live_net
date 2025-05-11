import torch
from torch.optim import Optimizer

class SignSGD(Optimizer):
    r"""Implements fixed‐step sign SGD:
        p ← p − lr * sign(∇p)
    Args:
        params (iterable): iterable of parameters to optimize or dicts
        lr (float): step size
    """

    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        loss = None
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                update = torch.sign(p.grad)
                p.add_( -lr * update )

        return loss
