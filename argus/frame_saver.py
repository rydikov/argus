import cv2
import logging
import os

from datetime import datetime


logger = logging.getLogger('json')


class FrameSaver:

    def __init__(self, sources_config):
        self.sources_config = sources_config

    def save(self, queue_elem, prefix=None):
        path = self.sources_config[queue_elem.thread_name]['stills_dir']
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        if not os.path.exists(path):
            os.makedirs(path)

        if prefix is None:
            frame_name = '{}.jpg'.format(timestamp)
        else:
            frame_name = '{}-{}.jpg'.format(timestamp, prefix)

        if not cv2.imwrite(os.path.join(path, frame_name), queue_elem.frame):
            logger.error('Unable to save file: %s' % frame_name)

        return '{}/{}'.format(self.sources_config[queue_elem.thread_name]['host_stills_uri'], frame_name)
