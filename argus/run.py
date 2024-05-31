import logging
import os

from logging import config

from argus.application import app

config.fileConfig(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        'logger/logger.conf'
    )
)

logger = logging.getLogger('json')

if __name__ == '__main__':
    app.run()
