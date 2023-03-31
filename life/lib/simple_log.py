import inspect
import sys
import os
import time
import re
import IPython


log_global_time = 0.0
log_ipython_execution_count = -1


def _get_code_info():
    frame1 = sys._getframe(1)
    frame2 = sys._getframe(2)
    return frame1.f_code.co_filename, frame2.f_code.co_filename, frame2.f_lineno


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
    this_filename, caller_filename, caller_lineno = _get_code_info()
    root_dir = os.path.dirname(this_filename) + "/.."
    file = caller_filename
    file = os.path.relpath(file, root_dir)
    line = caller_lineno
    location = f"{file}:{line}"
    is_pycharm_ipython_match = re.search(r"(\.\./)+tmp/ipykernel_\d+/\d+\.py$", file)
    if is_pycharm_ipython_match is not None:
        location = ""

    print(f"I\u02c8{time_:.3f}", *args, location)



