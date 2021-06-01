import logging

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
            'func:%r took: %2.4f sec' % (f.__name__, te - ts)
        )
        return result
    return wrap
