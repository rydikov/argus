import cv2
import logging
import sys

from time import sleep

from argus.utils.timing import timing
from argus.utils.bad_frame_checker import BadFrameChecker

RECONNECT_SLEEP_TIME = 1
logger = logging.getLogger('json')


class FrameGrabber:

    def __init__(self, config):

        try:
            self.cap = cv2.VideoCapture(config['source'])
        except Exception:
            logger.exception("Unable to create steam %s" % config['source'])
            self._exit()

        if not self.cap.isOpened():
            logger.error("Could not connect to camera: %s " % config['source'])
            self._exit()

        if 'bfc' in config:
            self.bfc = BadFrameChecker(config['bfc'])
        else:
            self.bfc = None

    def _exit(self):
        sleep(RECONNECT_SLEEP_TIME)
        sys.exit(1)

    @timing
    def make_snapshot(self):
        try:
            if self.cap.isOpened():
                __, frame = self.cap.read()
            else:
                logger.error("Cap is closed")
                self._exit()
        except Exception:
            logger.exception("Unable to get frame. Exit from thread")
            self._exit()

        if frame is None:
            logger.error('Empty frame. Exit from thread')
            self._exit()

        if self.bfc is not None and self.bfc.check(frame):
            return self.make_snapshot()

        return frame
