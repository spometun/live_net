import typing
from typing import Sequence, Iterable
import pytest
import torch

from life.lib.simple_log import LOG


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


def index_batcher(batch_size, epoch_size, provide_smaller_last=True):
    assert batch_size <= epoch_size
    i = 0
    while i + batch_size < epoch_size:
        yield i, i + batch_size
        i += batch_size
    if provide_smaller_last:
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

    batcher = index_batcher(2, 5, provide_smaller_last=False)
    values = [el for el in batcher]
    assert len(values) == 2
    assert values[0] == (0, 2)
    assert values[1] == (2, 4)


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


class BatchProvider:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size=1):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def batch(self):
        pass


def test_batch_provider():
    s = [1, 2]
    it = iter(s)
    LOG(next(it))
    LOG(next(it))
    LOG(next(it))


