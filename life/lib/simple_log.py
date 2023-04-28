import inspect
from enum import Enum
import sys
import os
import time
import re
import IPython


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1


level = LogLevel(1)


log_global_time = 0.0
log_ipython_execution_count = -1


def _get_code_info():
    frame1 = sys._getframe(1)
    frame3 = sys._getframe(3)
    return frame1.f_code.co_filename, frame3.f_code.co_filename, frame3.f_lineno


def _LOG(level, *args):
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

    print(f"{level}\u02c8{time_:.3f}", *args, location)


def LOG(*args):
    _LOG("I", *args)


def LOGD(*args):
    if level == LogLevel.DEBUG:
        _LOG("D", *args)


