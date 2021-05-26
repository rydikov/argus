import cv2
import logging
import pytesseract

from argus.helpers.timing import timing

logger = logging.getLogger(__file__)


@timing
def check_bad_frame(frame):
    h, w = frame.shape[0], frame.shape[1]

    # Day of week name box on frame
    y_min_percent = 4
    y_max_percent = 12.5
    x_min_percent = 26
    x_max_percent = 35

    y_min = round(h * y_min_percent / 100)
    y_max = round(h * y_max_percent / 100)
    x_min = round(w * x_min_percent / 100)
    x_max = round(w * x_max_percent / 100)

    frame = frame[y_min:y_max, x_min:x_max]

    def _recognize(frame):
        # psm 8: Treat the image as a single word.
        return pytesseract.image_to_string(frame, config='--psm 8').rstrip()
    
    recognized_day_of_week = _recognize(frame)
    if not bool(recognized_day_of_week):
	    frame = cv2.bitwise_not(frame) # Invert image
	    recognized_day_of_week = _recognize(frame)
    
    logger.info('Recognized day of week: {}'.format(recognized_day_of_week))

    return not bool(recognized_day_of_week)
