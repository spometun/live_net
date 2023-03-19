import inspect
import os
import time


log_global_time = 0.0


def LOG(*args):
    global log_global_time
    if log_global_time == 0.0:
        log_global_time = time.time()
    time_ = time.time() - log_global_time
    root_dir = os.path.dirname(inspect.stack()[0].filename) + "/../.."
    file = inspect.stack()[1].filename
    file = os.path.relpath(file, root_dir)
    line = inspect.stack()[1].lineno
    print(f"I\u02c8{time_:.6f}", *args, f"{file}:{line}")



