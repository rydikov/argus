import cv2
import logging
import sys

from time import sleep

from argus.helpers.timing import timing
from argus.helpers.bad_frame_checker import BadFrameChecker

logger = logging.getLogger('json')

SLEEP_TIME_IF_FRAME_IS_FAILED_SEC = 5
MAX_FAILED_FRAMES = 5


class FrameGrabber:

    def __init__(self, config):
        self.cap = cv2.VideoCapture(config['source'])
        self.failed_frames = 0

        if 'bfc' in config:
            self.bfc = BadFrameChecker(config['bfc'])
        else:
            self.bfc = None

    @timing
    def make_snapshot(self):
        __, frame = self.cap.read()

        if frame is None:
            logger.error('Unable to get frame')
            sleep(SLEEP_TIME_IF_FRAME_IS_FAILED_SEC)
            self.failed_frames += 1
            if self.failed_frames == MAX_FAILED_FRAMES:
                logger.error('Unable to get frames. Restart app')
                sys.exit(1)
            return self.make_snapshot()
        else:
            self.failed_frames = 0

        if self.bfc is not None and self.bfc.check(frame):
            return self.make_snapshot()

        return frame
