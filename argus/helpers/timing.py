import logging
import threading

from functools import wraps
from time import time

logger = logging.getLogger('json')


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
                'thread_name': threading.current_thread().name,
                'level': 'info'}
        )
        return result
    return wrap
