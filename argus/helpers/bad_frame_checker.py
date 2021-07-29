import cv2
import logging


logger = logging.getLogger(__file__)

MIN_WHITE = 50


class BadFrameChecker:
    def __init__(self, config):
        self.coords = config['coords']
        self.threshold = config['threshold']
        self.reverse_pixel = config.get('reverse_pixel')
        self.template = cv2.imread(
            config['template_path'],
            cv2.IMREAD_GRAYSCALE
        )

    def check(self, frame):

        frame = frame[
            self.coords[0]:self.coords[1],
            self.coords[2]:self.coords[3]
            ]

        # Checking reverse pixel. Checked digit may be black or white.
        # If background white – digit black, or on the contrary.
        if (
            self.reverse_pixel is not None
            and all(
                i > MIN_WHITE for i in frame[
                    self.reverse_pixel[0],
                    self.reverse_pixel[1]
                ]
            )
        ):
            frame = cv2.bitwise_not(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.matchTemplate(frame, self.template, cv2.TM_SQDIFF)

        if int(diff[0][0]) > self.threshold:
            logger.warning('Bad frame detected. Diff: {}'.format(diff))

        return int(diff[0][0]) > self.threshold
