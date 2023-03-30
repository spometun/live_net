import types
from typing import Union, Optional, Any
import numpy as np
import torch
from matplotlib import pyplot as plt
import scipy


class AccumStat:
    def __init__(self, value=None):
        self._values = []
        if value is not None:
            self.add_value(value)

    def add_value(self, value: Union[np.ndarray, torch.Tensor, Any]):
        if isinstance(value, str):
            return
        if isinstance(value, list) or \
           isinstance(value, tuple) or \
           isinstance(value, types.GeneratorType):
            for el in value:
                self.add_value(el)
            return

        try:
            value = value.detach()
        except AttributeError:
            pass
        self._values += list(value.flatten())

    def get_abs_max(self) -> float:
        vals = np.array(self._values)
        res = np.max(np.abs(vals))
        return res

    def get_amount(self) -> int:
        return len(self._values)

    def plot(self, name=None):
        if name is None:
            name = "Values plot3"
        plt.figure(figsize=(16, 9))
        plt.title(name)
        plt.plot(sorted(self._values))
        plt.grid()
        plt.show()


def get_slope_and_pvalue(values: list):
    y = np.array(values)
    x = np.arange(len(y))
    result = scipy.stats.linregress(x, y, alternative="less")
    return result.slope, result.pvalue


