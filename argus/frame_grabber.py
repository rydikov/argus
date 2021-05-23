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


class BadFrameChecker:
    
    def __init__(self):
        self.store_images = 5
        self.last_image_sizes = collections.deque([], self.store_images)
        self.deviation_percent = 60

    def is_image_size_less_avg_size(self, image_size):
        avg_image_size = sum(self.last_image_sizes)/self.store_images
        return (image_size/avg_image_size)*100 < self.deviation_percent

    def is_bad(self, image_path):
        image_size = os.path.getsize(image_path)
        self.last_image_sizes.appendleft(image_size)

        if len(self.last_image_sizes) < self.store_images:
            return False
        
        return self.is_image_size_less_avg_size(image_size)


class FrameGrabber:

    def __init__(self, config):
        self.config = config
        self.bfc = BadFrameChecker()
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

        if self.bfc.is_bad(snapshot_path):
            logger.warning('Bad file deleted')
            os.remove(snapshot_path)
            return
        
        return frame