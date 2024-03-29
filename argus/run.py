import logging
import os
import sys
import yaml

from logging import config

from argus.application import app

dir_path = os.path.dirname(os.path.realpath(__file__))
config.fileConfig(os.path.join(dir_path, 'logger/logger.conf'))

logger = logging.getLogger('json')


if __name__ == '__main__':

    try:
        config_path = sys.argv[1]
    except IndexError:
        logger.error('Configuration file path not specified')
        exit(1)

    with open(os.path.join(dir_path, config_path)) as f:
        config = yaml.safe_load(f)

    app.run(config)
