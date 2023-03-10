from typing import Union, Optional, Any
import numpy as np
from matplotlib import pyplot as plt


class AccumStat:
    def __init__(self, value=None):
        self._values = []
        if value is not None:
            self.add_value(value)

    def add_value(self, value: Union[np.ndarray, Any]):
        try:
            value = value.numpy()
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
        plt.title(name)
        plt.plot(sorted(self._values))
        plt.grid()
        plt.show()
