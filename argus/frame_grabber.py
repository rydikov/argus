import cv2
import logging
import sys

from time import sleep

from argus.helpers.timing import timing
from argus.helpers.bad_frame_checker import BadFrameChecker

logger = logging.getLogger('json')


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

        try:
            __, frame = self.cap.read()
        except Exception:
            logger.exception("Unable to get frame. Restart app")
            sys.exit(1)

        if frame is None:
            logger.error('Empty frame. Restart app')
            sys.exit(1)

        if self.bfc is not None and self.bfc.check(frame):
            return self.make_snapshot()

        return frame
