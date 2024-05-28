import logging
import os


logger = logging.getLogger('json')

class AlarmSystemService:

    def __init__(self):
        self.file_path = '/tmp/arming'

    def arming(self):
        with open(self.file_path, 'a'):
            os.utime(self.file_path, None)
        logger.info('Arming')

    def disarming(self):
        if self.is_armed():
            os.remove(self.file_path)
        logger.info('Disarming')

    def is_armed(self):
        return os.path.isfile(self.file_path)
