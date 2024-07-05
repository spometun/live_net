import threading


class LockedIterator:
    def __init__(self, iterator):
        self._lock = threading.Lock()
        self._iter = iter(iterator)  # support both iterables and iterators

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return next(self._iter)

