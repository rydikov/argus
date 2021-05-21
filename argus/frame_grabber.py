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

        if self.config['source'].startswith('rtsp'):
            self.snapshot_method = self.__make_rtsp_snapshot
        else:
            self.cap = cv2.VideoCapture(self.config['source'])
            self.snapshot_method = self.__make_video_snapshot

    @timing
    def __make_rtsp_snapshot(self, snapshot_path):
        stream = ffmpeg.input(self.config['source'], rtsp_transport='tcp', stimeout=DEADLINE_IN_MSEC)
        stream = stream.output(snapshot_path, vframes=1, pix_fmt='rgb24')
        try:
            stream.run(capture_stdout=True, capture_stderr=True)
        except ffmpeg._run.Error as e:
            logger.exception("Time out Error")
            raise
        
    @timing
    def __make_video_snapshot(self, snapshot_path):
        __, frame = self.cap.read()
        is_saved = cv2.imwrite(snapshot_path, frame)
        if not is_saved:
            raise

    def make_snapshot(self):

        snapshot_path = "{}/{}.jpg".format(self.config['stills_dir'], datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))

        try:
            self.snapshot_method(snapshot_path)
        except:
            logger.exception("Unable to make snapshot")
            return
        
        if self.bfc.is_bad(snapshot_path):
            logger.warning('Bad file deleted')
            os.remove(snapshot_path)
            return
        
        return snapshot_path