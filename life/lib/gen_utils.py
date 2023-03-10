import typing


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
