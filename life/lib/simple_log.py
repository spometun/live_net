import inspect
import os
import time
import re
import IPython


log_global_time = 0.0
log_ipython_execution_count = -1


def LOG(*args):
    global log_global_time
    global log_ipython_execution_count

    ipython = IPython.get_ipython()
    if ipython is not None:
        cur_count = IPython.get_ipython().execution_count
        if cur_count != log_ipython_execution_count:
            log_global_time = 0.0
            log_ipython_execution_count = cur_count

    if log_global_time == 0.0:
        log_global_time = time.time()

    time_ = time.time() - log_global_time
    root_dir = os.path.dirname(inspect.stack()[0].filename) + "/../.."
    file = inspect.stack()[1].filename
    file = os.path.relpath(file, root_dir)
    line = inspect.stack()[1].lineno
    location = f"{file}:{line}"
    is_pycharm_ipython_match = re.search(r"(\.\./)+tmp/ipykernel_\d+/\d+\.py$", file)
    if is_pycharm_ipython_match is not None:
        location = ""

    print(f"I\u02c8{time_:.3f}", *args, location)



