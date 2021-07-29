import cv2
import logging
import os

from datetime import datetime


logger = logging.getLogger(__file__)


class FrameSaver:

    def __init__(self, config):
        self.stills_dir = config['stills_dir']
        self.host_stills_uri = config['host_stills_uri']

    def save(self, frame, prefix=None):
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        if not os.path.exists(self.stills_dir):
            os.makedirs(self.stills_dir)

        if prefix is None:
            file_name = '{}.jpg'.format(timestamp)
        else:
            file_name = '{}-{}.jpg'.format(timestamp, prefix)

        file_path = os.path.join(self.stills_dir, file_name)
        is_saved = cv2.imwrite(file_path, frame)
        
        if not is_saved:
            logger.error('Unable to save file. Prefix: %s' % prefix)
        
        return '{}/{}'.format(self.host_stills_uri, file_name)