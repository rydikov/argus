import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

from argus.globals import aqara_service

aqara_service.run_scene()