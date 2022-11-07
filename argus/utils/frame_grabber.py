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

        self.snapshot_delay = config.get('snapshot_delay')
        self.source = config['source']

        if 'bfc' in config:
            self.bfc = BadFrameChecker(config['bfc'])
        else:
            self.bfc = None

    @timing
    def make_snapshot(self):

        cap = cv2.VideoCapture(self.source)
        
        if self.snapshot_delay:
            sleep(self.snapshot_delay)

        try:
            if cap.isOpened():
                __, frame = cap.read()
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
        
        cap.release()

        return frame

    def _exit(self):
        sleep(RECONNECT_SLEEP_TIME)
        sys.exit(1)

