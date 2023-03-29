from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import life.lib as lib


if __name__ == "__main__":
    x = np.arange(20)
    y = x * x
    plt.ion()
    criterion = lib.nets.criterion_n
    net = lib.nets.PERCEPTRON(784, 2)
    # net = lib.livenet.LiveNet(784, None, 2)
    sizes, times = lib.utils.calc_batch_times(net, criterion, 4096)
    plt.plot(sizes, times)
    plt.grid()
    pass