import logging
import threading

from pythonjsonlogger import jsonlogger
from functools import wraps
from time import time

logger = logging.getLogger(__file__)

formatter = jsonlogger.JsonFormatter()
logHandler = logging.StreamHandler()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)


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
            ),
            extra = {
                'time': te-ts, 
                'func': f.__name__,
                'thread_name': threading.current_thread().name}
        )
        return result
    return wrap
