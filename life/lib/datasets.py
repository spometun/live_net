import torch
import torchvision


def get_xor():
    xor_x = torch.Tensor([[0., 0.],
                          [0., 1.],
                          [1., 0.],
                          [1., 1.]])
    xor_y = torch.tensor([0, 1, 1, 0]).reshape(xor_x.shape[0], 1)
    return xor_x, xor_y


def get_odd():
    odd_x = torch.Tensor([[0.], [1.], [2.]])
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


def get_linear3():
    x = torch.tensor([[1, 0, 0.8],
                      [0, 1, 0.1]
                      ])
    y = torch.tensor([[0], [1]])
    return x, y


def _get_mnist(train: bool):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()
            , torchvision.transforms.Normalize([0.5], [0.5])
         ])
    dataset=torchvision.datasets.MNIST("/home/spometun/datasets/research", train=train,
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


if __name__ == "__main__":
    t = get_mnist_test()
    pass

