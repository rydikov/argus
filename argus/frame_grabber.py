import cv2
import logging

from time import time

from argus.helpers.bad_frame_checker import BadFrameChecker

logger = logging.getLogger(__file__)

MAX_SNAPSHOT_DELAY_SEC = 5


class FrameGrabber:

    def __init__(self, config):
        self.cap = cv2.VideoCapture(config['source'])

        if 'bfc' in config:
            self.bfc = BadFrameChecker(config['bfc'])
        else:
            self.bfc = None

    def make_snapshot(self):
        ts = time()
        __, frame = self.cap.read()
        te = time()

        if te - ts > MAX_SNAPSHOT_DELAY_SEC:
            logger.warning('snapshot delay: %2.4f sec' % (te - ts))

        if self.bfc is not None and self.bfc.check(frame):
            logger.warning('Bad frame detected. Ignored.')
            self.make_snapshot()

        return frame
