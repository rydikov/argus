import logging
import os

from logging import config

from argus.application import app

dir_path = os.path.dirname(os.path.realpath(__file__))
config.fileConfig(os.path.join(dir_path, 'logger/logger.conf'))

logger = logging.getLogger('json')

if __name__ == '__main__':
    app.run()
