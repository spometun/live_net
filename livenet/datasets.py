import numpy as np
import torch
import torchvision
from scipy import ndimage
import os

datasets_dir = f"{os.getenv('HOME')}/datasets/research"


def permutate_sync(data1, data2, rng=None):
    assert len(data1) == len(data2)
    if rng is None:
        rng = np.random.default_rng(0)
    permutation = rng.permutation(len(data1))
    data1_perm = data1[permutation]
    data2_perm = data2[permutation]
    return data1_perm, data2_perm


def augment_vertical_flip(data, labels):
    flipped = torch.flip(data, dims=[-1])
    augmented_x = torch.concatenate((data, flipped), dim=0)
    augmented_y = torch.concatenate((labels, labels), dim=0)
    augmented_x, augmented_y = permutate_sync(augmented_x, augmented_y)
    return augmented_x, augmented_y


def augment_ten_rotation_shifts(data, labels, rotation_degrees: float, crop_pixels: int):
    data = np.array(data.cpu())
    labels = np.array(labels.cpu())
    augmented_x = get_ten_rotation_shifts(data, rotation_degrees, crop_pixels)
    augmented_x = np.concatenate((data, augmented_x), axis=0)
    augmented_y = np.concatenate(11 * [labels], axis=0)
    augmented_x, augmented_y = permutate_sync(augmented_x, augmented_y)
    augmented_x = torch.tensor(augmented_x, device="cpu")
    augmented_y = torch.tensor(augmented_y, device="cpu")
    return augmented_x, augmented_y


def get_ten_rotation_shifts(data: np.ndarray, rotation_degrees: float, crop_pixels: int):
    assert data.ndim == 4 # assumes input is nchw
    augmented = []

    for angle in [-rotation_degrees, rotation_degrees]:
        r = ndimage.rotate(data, angle, axes=(3, 2))
        r = r[:, :, 1:-1,1:-1]
        c = crop_pixels
        augmented.append(r[:, :, :-c, :-c])
        augmented.append(r[:, :, c:, :-c])
        augmented.append(r[:, :, c:, c:])
        augmented.append(r[:, :, :-c, c:])
        augmented.append(r[:, :, c // 2:-c // 2, c // 2:-c // 2])

    result = np.concatenate(augmented, axis=0)
    return result


def get_xor():
    xor_x = torch.Tensor([[0., 0.],
                          [0., 1.],
                          [1., 0.],
                          [1., 1.]])
    xor_y = torch.tensor([0, 1, 1, 0]).reshape(xor_x.shape[0], 1)
    return xor_x, xor_y


def get_odd_2():
    odd_x = torch.tensor([[0.], [1.], [2.]])
    odd_y = torch.tensor([[0], [1], [0]])
    return odd_x, odd_y


def get_pyramid():
    x = torch.Tensor([[0., 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0],
                      [1, 1, 1]])
    y = torch.tensor([[1], [0], [0], [0], [1]])
    return x, y


def get_linear2():
    x = torch.tensor([[1., 0],
                      [0, 1]
                      ])
    y = torch.tensor([[0], [1]])
    return x, y


def get_linear3():
    x = torch.tensor([[1, 0, 1],
                      [0, 1, 0.]
                      ])
    y = torch.tensor([[0], [1]])
    return x, y


def _get_mnist(train: bool):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()
            , torchvision.transforms.Normalize([0.5], [0.5])
         ])
    dataset=torchvision.datasets.MNIST(datasets_dir, train=train,
                                    download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    whole = next(iter(loader))
    data = whole[0]
    labels = whole[1]
    labels = labels[:, None]
    return data, labels


def get_mnist_test():
    return _get_mnist(False)


def get_mnist_train():
    return _get_mnist(True)


def _get_cifar10(train: bool):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()
            , torchvision.transforms.Normalize([0.5], [0.5])
         ])
    dataset = torchvision.datasets.CIFAR10(datasets_dir, train=train,
                                       download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    whole = next(iter(loader))
    data = whole[0]
    labels = whole[1]
    labels = labels[:, None]
    return data, labels


def get_cifar10_test():
    return _get_cifar10(False)


def get_cifar10_train():
    data, labels = _get_cifar10(True)
    return data, labels


def to_plain(x, y, downscale=1, to_odd=False, to_gray=False):
    if not isinstance(downscale, tuple):
        d = (downscale, downscale)
    else:
        d = downscale
    n0 = x.shape[2]
    n1 = x.shape[3]
    assert n0 % d[0] == 0 and n1 % d[1] == 0
    x = x.reshape(x.shape[0], x.shape[1], n0 // d[0], d[0], n1 // d[1], d[1])
    x = x.mean(axis=(3, 5), keepdims=False)
    if to_gray:
        x = x.mean(axis=1, keepdims=False)
    x = x.reshape(len(x), -1)
    if to_odd:
        y = y % 2
    return x, y


if __name__ == "__main__":
    t = get_cifar10_train()
    pass

