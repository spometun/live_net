from functools import reduce
from typing import Iterable, Callable, Any


# Chain Callables
# For example, given f = chain(f1, f2, f3)
# f(x) is equivalent to f1(f2(f3(x)))
def func_chain(*functions: Iterable[Callable[[Any], Any]]) -> Callable[[Any], Any]:
    return reduce(lambda f, g: lambda x: f(g(x)), functions)
