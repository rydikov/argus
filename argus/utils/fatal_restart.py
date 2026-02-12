import os
import logging


logger = logging.getLogger('json')


def fatal_restart(reason: str, code: int = 1) -> None:
    logger.critical('Fatal restart required: %s', reason)
    os._exit(code)
