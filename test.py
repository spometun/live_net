print(f"package {__package__}")
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
#from simple_log import LOG
from . import livenet
core = livenet.core


class C:
    def __init__(self):
        self.a1 = 9


def f():
    c = C()
    LOG(f"{c.a2}")


def test_vs():
    #plt.ion()
    print("hivs")
    a = 3
    print(a * 2)


if __name__ == "__main__":
    x = np.arange(20)
    y = x * x
    # plt.ion()
    criterion = core.nets.criterion_n
    net = core.nets.PERCEPTRON(784, 2)
    # net = core.livenet.LiveNet(784, None, 2)
    sizes, times = core.utils.calc_batch_times(net, criterion, 4096)
    # plt.plot(sizes, times)
    # plt.grid()
    pass