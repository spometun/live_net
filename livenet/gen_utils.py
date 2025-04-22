import typing
from typing import Sequence, Iterable
import pytest
import torch
import numpy as np

from ai_libs.simple_log import LOG


class IndexTakerGenerator:
    def __init__(self, data, index):
        self._data = data
        self._index = index

    def __iter__(self):
        class _IndexTaker:
            def __init__(self, parent):
                self._iter = iter(parent._data)
                self._parent = parent

            def __next__(self):
                return next(self._iter)[self._parent._index]

            def __len__(self):
                return self._parent.__len__()

            def __iter__(self):
                return self

        return _IndexTaker(self)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item][self._index]


def split_dataset(dataset) -> tuple:
    data = IndexTakerGenerator(dataset, 0)
    labels = IndexTakerGenerator(dataset, 1)
    return data, labels


class ZipWithLen:
    def __init__(self, a: Sequence, b: Sequence):
        assert len(a) == len(b)
        self.a = a
        self.b = b

    def __iter__(self):
        for i in range(len(self.a)):
            yield self.a[i], self.b[i]

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i]


def index_batcher(batch_size, epoch_size, skip_smaller_last=False):
    assert batch_size <= epoch_size
    i = 0
    while i + batch_size < epoch_size:
        yield i, i + batch_size
        i += batch_size
    if not skip_smaller_last or i + batch_size == epoch_size:
        yield i, epoch_size


def test_index_batcher():
    batcher = index_batcher(2, 5)
    values = [el for el in batcher]
    assert values[0] == (0, 2)
    assert values[1] == (2, 4)
    assert values[2] == (4, 5)

    batcher = index_batcher(2, 2)
    values = [el for el in batcher]
    assert len(values) == 1
    assert values[0] == (0, 2)

    batcher = index_batcher(3, 6)
    values = [el for el in batcher]
    assert len(values) == 2
    assert values[0] == (0, 3)
    assert values[1] == (3, 6)

    batcher = index_batcher(2, 5, skip_smaller_last=True)
    values = [el for el in batcher]
    assert len(values) == 2
    assert values[0] == (0, 2)
    assert values[1] == (2, 4)

    batcher = index_batcher(2, 6, skip_smaller_last=True)
    values = [el for el in batcher]
    assert len(values) == 3


def test_zip_with_len():
    a = ["a", "b", "c", "d", "e"]
    b = [1, 2, 3, 4, 5]
    both = ZipWithLen(a, b)
    n = 0
    for _ in both:
        n += 1
    for _ in both:
        n += 1
    assert n == 2 * len(a)


def batch_iterator(x: np.ndarray, y: np.ndarray, batch_size=1, skip_smaller_last=True, only_one_epoch=False, seed=0):
    assert len(x) == len(y)
    epoch_size = len(x)
    rnd = None
    x = x.copy()
    y = y.copy()
    while True:
        for start, end in index_batcher(batch_size, epoch_size, skip_smaller_last):
            yield x[start:end], y[start:end]
        if only_one_epoch:
            break
        if rnd is None:
            rnd = np.random.default_rng(seed=seed)
        permutation = rnd.permutation(epoch_size)
        x = x[permutation]
        y = y[permutation]


def test_batch_iterator():
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    y = torch.tensor([1, 0, 1, 0, 1, 0])
    it = batch_iterator(x, y, 2)
    for i in range(10):
        val = next(it)
        assert len(val) == 2
        assert (val[0][0] % 2 == val[1][0])
        assert (val[0][1] % 2 == val[1][1])





