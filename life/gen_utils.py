import typing


class IndexTakerGenerator:
    def __init__(self, data, index):
        self._data = data
        self._index = index

    def __iter__(self):
        class _IndexTaker:
            def __init__(self, data, index):
                self._data = data
                self._index = index
                self._cur = -1

            def __next__(self):
                self._cur += 1
                if self._cur >= len(self._data):
                    raise StopIteration
                return self[self._cur]

            def __getitem__(self, item):
                return self._data[item][self._index]

            def __len__(self):
                return len(self._data)

            def __iter__(self):
                return self

        return _IndexTaker(self._data, self._index)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item][self._index]


def split_dataset(dataset) -> tuple:
    data = IndexTakerGenerator(dataset, 0)
    labels = IndexTakerGenerator(dataset, 1)
    return data, labels
