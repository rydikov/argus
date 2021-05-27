import os
import cv2
import logging


from argus.helpers.timing import timing

logger = logging.getLogger(__file__)

THRESHOLD = 10000000


class BadFrameChecker:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.template = cv2.imread(os.path.join(dir_path, '../../res/2.jpg'), cv2.IMREAD_GRAYSCALE)

    @timing
    def check(self, frame):
        """ Values for 1920x1080 Image """

        frame = frame[64:104, 324:352]

        # digit is white
        if frame[39, 0][0] > 50:
            frame = cv2.bitwise_not(frame)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.matchTemplate(frame, self.template, cv2.TM_SQDIFF)
        
        if int(diff[0][0]) > THRESHOLD:
            logger.warning('Diff: {}'.format(diff))
        
        
        return int(diff[0][0]) > THRESHOLD
