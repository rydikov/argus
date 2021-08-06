import logging
import os
import sys

import yaml

from argus import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


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
