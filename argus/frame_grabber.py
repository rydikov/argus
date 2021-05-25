import cv2
import ffmpeg
import logging
import os 
import collections

from datetime import datetime

from argus.helpers.timing import timing

# 25 sec
DEADLINE_IN_MSEC = 25000000

logger = logging.getLogger(__file__)


class FrameGrabber:

    def __init__(self, config):
        self.config = config
        self.cap = cv2.VideoCapture(self.config['source'])

    @timing
    def make_snapshot(self):
        snapshot_path = "{}/{}.jpg".format(self.config['stills_dir'], datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))

        __, frame = self.cap.read()
        if frame is None:
            return

        is_saved = cv2.imwrite(snapshot_path, frame)
        if not is_saved:
            return

        return frame