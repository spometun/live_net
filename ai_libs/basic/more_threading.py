import threading
from typing import Callable, Iterator, Any
from .even_more_itertools import LockedIterator


def execute_threaded(func: Callable[[Any], Any], iterator: Iterator[Any], n_threads: int):
    #  calls 'func' with every value from 'iterator' in parallel using 'n_threads' threads
    assert n_threads >= 1, f"Invalid amount of threads requested: {n_threads}"
    iter = LockedIterator(iterator)
    def thread_func():
        for element in iter:
            func(element)
    if n_threads == 1:
        thread_func()
    else:
        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=thread_func)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
