import cv2
import logging
import pytesseract

from datetime import datetime
from argus.helpers.timing import timing

logger = logging.getLogger(__file__)


@timing
def check_bad_frame(frame):
    """ Values for 1920x1080 Image """
    
    day_of_week = datetime.today().strftime('%a')

    # pixel from first digit on year on text
    is_white = frame[80, 378][0] > 15

    # Day of week name box on frame
    frame = frame[43:135, 499:672]

    if is_white:
        # if text is white, reverse image
	    frame = cv2.bitwise_not(frame)

    # psm 8: Treat the image as a single word.
    recognized_day_of_week = pytesseract.image_to_string(frame, config='--psm 8').rstrip()
    
    logger.info('Recognized day of week: {}'.format(recognized_day_of_week))

    return day_of_week != recognized_day_of_week 
