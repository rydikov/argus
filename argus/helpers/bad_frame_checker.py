import cv2
import logging


logger = logging.getLogger('json')

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

        subframe = frame[
            self.coords[0]:self.coords[1],
            self.coords[2]:self.coords[3]
            ]

        # Checking reverse pixel. Checked digit may be black or white.
        # If background white – digit black, or on the contrary.
        if (
            self.reverse_pixel is not None
            and all(
                i > MIN_WHITE for i in subframe[
                    self.reverse_pixel[0],
                    self.reverse_pixel[1]
                ]
            )
        ):
            subframe = cv2.bitwise_not(subframe)

        subframe = cv2.cvtColor(subframe, cv2.COLOR_BGR2GRAY)

        diff = int(cv2.matchTemplate(subframe, self.template, cv2.TM_SQDIFF)[0][0])

        # Mark diff on frame for analize
        cv2.putText(
            frame,
            f"{diff:,}",
            (20, 20),  # position
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 0, 255),  # red
            1
        )

        if diff > self.threshold:
            logger.warning('Bad frame detected. Diff: {}'.format(diff))
            return True

        return False
