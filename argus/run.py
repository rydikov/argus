import logging
import os
import sys
import yaml

from logging import config

from argus import app


config.fileConfig('log.conf')

logger = logging.getLogger('json')


if __name__ == '__main__':

    try:
        config_path = sys.argv[1]
    except IndexError:
        logger.error('Configuration file path not specified')
        exit(1)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, config_path)) as f:
        config = yaml.safe_load(f)

    app.run(config)
