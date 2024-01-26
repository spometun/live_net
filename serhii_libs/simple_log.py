import inspect
from enum import Enum
import sys
import os
import time
import re
import logging
from io import StringIO


logger = logging.getLogger("simple_log")
log_to_logger = False
log_to_console = True


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


level = LogLevel(1)


log_global_time = time.time()
log_ipython_execution_count = -1


class bcolors:
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def _init():
    global log_global_time
    global log_ipython_execution_count
    try:
        import IPython
    except ModuleNotFoundError:
        pass
    else:
        ipython = IPython.get_ipython()
        if ipython is not None:
            cur_count = IPython.get_ipython().execution_count
            if cur_count != log_ipython_execution_count:
                log_global_time = time.time()
                log_ipython_execution_count = cur_count


_init()


def _get_code_info():
    frame1 = sys._getframe(1)
    frame3 = sys._getframe(3)
    return frame1.f_code.co_filename, frame3.f_code.co_filename, frame3.f_lineno


def _LOG(level, *args):
    _init()
    global log_global_time
    # global log_ipython_execution_count

    # if log_global_time == 0.0:
    #     log_global_time = time.time()

    time_ = time.time() - log_global_time
    this_filename, caller_filename, caller_lineno = _get_code_info()
    root_dir = os.getcwd()
    file = caller_filename
    file = os.path.relpath(file, root_dir)
    line = caller_lineno
    location = f"{file}:{line}"
    is_pycharm_ipython_match = re.search(r"(\.\./)+tmp/ipykernel_\d+/\d+\.py$", file)
    if is_pycharm_ipython_match is not None:
        location = ""

    s = StringIO()
    print(f"{level}\u02c8{time_:.3f}", *args, location, end="", file=s)
    s = s.getvalue()

    if log_to_console:
        if level == "D" or level == "I":
            print(s, file=sys.stdout)
        elif level == "W":
            print(f"{bcolors.WARNING}{s}{bcolors.ENDC}", file=sys.stdout)
        else:
            print(f"{bcolors.ERROR}{s}{bcolors.ENDC}", file=sys.stderr)

    if log_to_logger:
        if level == "D":
            logger.debug(s)
        if level == "I":
            logger.info(s)
        if level == "W":
            logger.warning(s)
        if level == "E":
            logger.error(s)


def LOGD(*args):
    if level == LogLevel.DEBUG:
        _LOG("D", *args)


def LOG(*args):
    if level == LogLevel.DEBUG or level == LogLevel.INFO:
        _LOG("I", *args)


def LOGW(*args):
    if level == LogLevel.DEBUG or level == LogLevel.INFO or level == LogLevel.WARNING:
        _LOG("W", *args)


def LOGE(*args):
        _LOG("E", *args)
        exit(-1)
