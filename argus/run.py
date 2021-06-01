import logging
import os

import yaml

from argus import app

MODE = os.environ.get('MODE', 'development')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S'
)


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '../conf/app.yml')) as f:
        config = yaml.safe_load(f)[MODE]

    app.run(config, MODE)
