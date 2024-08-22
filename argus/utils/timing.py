import logging

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
            'func:%r took: %2.4f sec' % (
                f.__name__,
                te - ts
            ),
            extra={
                'leadTime': te-ts,
                'func': f.__name__,
            }
        )
        return result
    return wrap



class Throttler:
    def __init__(self):
        self.last_time_called = None

    def is_allowed(self, interval):
        now = time()
        if self.last_time_called is None or now - self.last_time_called >= interval:
            self.last_time_called = now
            return True
        return False