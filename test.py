print(f"package {__package__}")
import numpy as np
from ai_libs.simple_log import LOG
from . import livenet
core = livenet.core


class N:
    def __init__(self, c):
        LOG("N", type(self), c)
        self.c = c


class S(N):
    def __init__(self, c, c2):
        LOG("S", type(self), c)
        super().__init__(c)
        LOG("after super S")
        self.c = c


class D(N):
    def __init__(self, c):
        LOG("D", type(self), c)
        super().__init__(c)
        LOG("after super D")
        self.c = c


class R(D, S):
    def __init__(self, c):
        LOG("R", type(self), c)
        super().__init__(c)
        # super(D, self).__init__(p3)
        # super(D, self).__init__(p3)


def f():
    r = R(3)
    LOG(R.mro())


if __name__ == "__main__":
    # f()
    LOG(livenet.core.livenet.RegularNeuron.mro())
    exit(0)
    x = np.arange(20)
    y = x * x
    # plt.ion()
    criterion = livenet.nets.criterion_classification_n
    net = livenet.nets.PERCEPTRON(784, 2)
    # net = core.livenet.LiveNet(784, None, 2)
    sizes, times = livenet.utils.calc_batch_times(net, criterion, 4096)
    # plt.plot(sizes, times)
    # plt.grid()
    pass