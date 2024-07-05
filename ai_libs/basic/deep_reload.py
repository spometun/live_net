from types import ModuleType
import sys
import importlib


def deep_reload(m: ModuleType):
    name = m.__name__  # get the name that is used in sys.modules
    name_ext = name + '.'  # support finding sub modules or packages

    def compare(loaded: str):
        return (loaded == name) or loaded.startswith(name_ext)

    all_mods = tuple(sys.modules)  # prevent changing iterable while iterating over it
    sub_mods = filter(compare, all_mods)
    for pkg in sorted(sub_mods, key=lambda item: item.count('.'), reverse=True):
        # print(f"reloading {sys.modules[pkg]}")
        importlib.reload(sys.modules[pkg])  # reload packages, beginning with the most deeply nested

