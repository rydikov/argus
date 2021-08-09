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
            'func:%r in thread %s took: %2.4f sec' % (
                f.__name__,
                te - ts
            ),
            extra = {
                'leadTime': te-ts, 
                'func': f.__name__,
            }
        )
        return result
    return wrap
