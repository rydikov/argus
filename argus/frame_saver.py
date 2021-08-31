import cv2
import logging
import os

from datetime import datetime


logger = logging.getLogger('json')


class FrameSaver:

    def __init__(self, config):
        self.stills_dir = config['stills_dir']
        self.host_stills_uri = config['host_stills_uri']
        self.save_every_n_frame = config.get('save_every_n_frame', 0)
        self.frame_count = 0

    def save(self, frame, prefix):

        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        if not os.path.exists(self.stills_dir):
            os.makedirs(self.stills_dir)

        if prefix is None:
            frame_name = '{}.jpg'.format(timestamp)
        else:
            frame_name = '{}-{}.jpg'.format(timestamp, prefix)

        file_path = os.path.join(self.stills_dir, frame_name)

        if not cv2.imwrite(file_path, frame):
            logger.error('Unable to save file: %s' % frame_name)

        return '{}/{}'.format(self.host_stills_uri, frame_name)


    def save_if_need(self, frame, forced=False, prefix=None):
        self.frame_count += 1
        if self.frame_count == self.save_every_n_frame or forced:
            self.frame_count = 0
            return self.save(frame, prefix)

