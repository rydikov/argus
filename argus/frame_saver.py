import cv2
import logging
import os

from datetime import datetime


logger = logging.getLogger('json')


class FrameSaver:

    @staticmethod
    def save(frame, path, prefix=None):

        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        if not os.path.exists(path):
            os.makedirs(path)

        if prefix is None:
            frame_name = '{}.jpg'.format(timestamp)
        else:
            frame_name = '{}-{}.jpg'.format(timestamp, prefix)

        if not cv2.imwrite(os.path.join(path, frame_name), frame):
            logger.error('Unable to save file: %s' % frame_name)

        return frame_name
