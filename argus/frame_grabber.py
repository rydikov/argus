import cv2
import logging
import sys

from time import time

from argus.helpers.timing import timing
from argus.helpers.bad_frame_checker import BadFrameChecker

logger = logging.getLogger('json')

MAX_SNAPSHOT_DELAY_SEC = 5


class FrameGrabber:

    def __init__(self, config):
        self.cap = cv2.VideoCapture(config['source'])

        if 'bfc' in config:
            self.bfc = BadFrameChecker(config['bfc'])
        else:
            self.bfc = None

    @timing
    def make_snapshot(self):
        ts = time()
        __, frame = self.cap.read()
        te = time()

        if te - ts > MAX_SNAPSHOT_DELAY_SEC:
            logger.warning('snapshot delay: %2.4f sec' % (te - ts))

        if frame is None:
            logger.error("Unable to get frame")
            sys.exit(1)

        if self.bfc is not None and self.bfc.check(frame):
            return self.make_snapshot()

        return frame
