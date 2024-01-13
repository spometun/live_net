import sys
print(f"path {sys.path}")
import importlib

from . import gen_utils
importlib.reload(gen_utils)

from . import graph
importlib.reload(graph)

from . import stat_utils
importlib.reload(stat_utils)

from . import optimizers
importlib.reload(optimizers)

from . import visual_utils
importlib.reload(visual_utils)

from . import utils
importlib.reload(utils)

from . import death
importlib.reload(death)

from . import livenet
importlib.reload(livenet)

from . import datasets
importlib.reload(datasets)

from . import nets
importlib.reload(nets)

from . import net_trainer
importlib.reload(net_trainer)

from . import test_livenet
importlib.reload(test_livenet)


