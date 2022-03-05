# Standard library imports
import functools
import logging
import os
import sys
import time
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


def _inputfile_exists(f):
    """Function decorator for checking whether an input file (first argument) exists."""

    def wrapper(*args, **kwargs):
        filename = args[0]
        if not Path(filename).exists():
            logger.warning("Input file does not exist. [{}]".format(filename))
        return f(*args, **kwargs)

    return wrapper


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        logger = logging.getLogger(os.path.basename(func.__module__))
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        logger.debug("Finished %s in %.4f secs", func.__name__, run_time)
        return value

    return wrapper_timer


def setup_logging():
    """Setups the initial logging configuration (e.g., stdout, format, warning filtering)."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s | %(message)s')
    handler.setFormatter(format)
    logger.addHandler(handler)

    warnings.filterwarnings("ignore")
