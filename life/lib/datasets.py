import torch


def get_xor():
    xor_x = torch.Tensor([[0., 0.],
                   [0., 1.],
                   [1., 0.],
                   [1., 1.]])

    xor_y = torch.Tensor([0., 1., 1., 0.]).reshape(xor_x.shape[0], 1)
    return xor_x, xor_y


def get_odd():
    odd_x = torch.Tensor([[0.], [1.], [2.]])
    odd_y = torch.Tensor([[0.], [1.], [0.]])
    return odd_x, odd_y


def get_pyramid():
    x = torch.Tensor([[0., 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 1, 1]])
    y = torch.Tensor([[1], [0], [0], [0], [1]])
    return x, y
