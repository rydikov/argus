import cv2
import logging

from argus.helpers.timing import timing
from argus.helpers.bad_frame_checker import BadFrameChecker

logger = logging.getLogger(__file__)


class FrameGrabber:

    def __init__(self, config):
        self.cap = cv2.VideoCapture(config['source'])

        if 'bfc' in config:
            self.bfc = BadFrameChecker(config['bfc'])
        else:
            self.bfc = None

    @timing
    def make_snapshot(self):
        __, frame = self.cap.read()

        if self.bfc is not None and self.bfc.check(frame):
            logger.warning('Bad frame detected. Ignored.')
            self.make_snapshot()

        return frame
