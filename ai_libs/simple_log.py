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
    frame1 = sys._getframe(2)
    frame3 = sys._getframe(4)
    return frame1.f_code.co_filename, frame3.f_code.co_filename, frame3.f_lineno


def _get_log_text(*args):
    _init()
    global log_global_time

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

    text = StringIO()
    print(f"\u02c8{time_:.3f}", *args, location, end="", file=text)
    text = text.getvalue()
    return text


def get_log_text(*args):
    # ensure function call depth the same as if call LOG()
    def f():
        return _get_log_text(*args)
    return f()


def _LOG(level, *args):
    text = _get_log_text(*args)
    text = f"{level}{text}"

    if log_to_console:
        if level == "D" or level == "I":
            print(text, file=sys.stdout)
        elif level == "W":
            print(f"{bcolors.WARNING}{text}{bcolors.ENDC}", file=sys.stdout)
        else:
            print(f"{bcolors.ERROR}{text}{bcolors.ENDC}", file=sys.stderr)

    if log_to_logger:
        if level == "D":
            logger.debug(text)
        if level == "I":
            logger.info(text)
        if level == "W":
            logger.warning(text)
        if level == "E":
            logger.error(text)


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
