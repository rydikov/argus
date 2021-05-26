import cv2
import logging

from argus.helpers.timing import timing

logger = logging.getLogger(__file__)


class FrameGrabber:

    def __init__(self, config):
        self.config = config
        self.cap = cv2.VideoCapture(self.config['source'])

    @timing
    def make_snapshot(self):
        __, frame = self.cap.read()
        return frame