import cv2
import logging
import sys

from time import sleep
from argus.helpers.timing import timing
from argus.helpers.bad_frame_checker import BadFrameChecker

RECONNECT_SLEEP_TIME = 1
logger = logging.getLogger('json')


class FrameGrabber:

    def __init__(self, config):

        self.cap = cv2.VideoCapture(config['source'])

        if not self.cap.isOpened():
            raise Exception("Could not connect to camera: %s " % config['source'])

        if 'bfc' in config:
            self.bfc = BadFrameChecker(config['bfc'])
        else:
            self.bfc = None

    @timing
    def make_snapshot(self):
        try:
            __, frame = self.cap.read()
        except Exception:
            logger.exception("Unable to get frame. Exit from thread")
            sleep(RECONNECT_SLEEP_TIME)
            sys.exit(1)

        if frame is None:
            logger.error('Empty frame. Exit from thread')
            sleep(RECONNECT_SLEEP_TIME)
            sys.exit(1)

        if self.bfc is not None and self.bfc.check(frame):
            return self.make_snapshot()

        return frame
