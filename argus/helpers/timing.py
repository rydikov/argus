import logging
import threading

from functools import wraps
from time import time


logger = logging.getLogger(__file__)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info(
            'func:%r in thread %s took: %2.4f sec' % (
                f.__name__,
                threading.current_thread().name,
                te - ts
            )
        )
        return result
    return wrap
