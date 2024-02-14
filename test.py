print(f"package {__package__}")
import numpy as np
#from simple_log import LOG
from . import livenet
core = livenet.core


if __name__ == "__main__":
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