import logging
import pytesseract

from datetime import datetime
from argus.helpers.timing import timing

logger = logging.getLogger(__file__)


@timing
def check_bad_frame(frame):
    h, w = frame.shape[0], frame.shape[1]

    # Year box on frame
    y_min_percent = 4
    y_max_percent = 12.5
    x_min_percent = 16
    x_max_percent = 26

    y_min = round(h * y_min_percent / 100)
    y_max = round(h * y_max_percent / 100)
    x_min = round(w * x_min_percent / 100)
    x_max = round(w * x_max_percent / 100)

    frame = frame[y_min:y_max, x_min:x_max]

    # psm 8: Treat the image as a single word.
    recognized_year = pytesseract.image_to_string(
        frame, 
        config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
    ).rstrip()
    logger.info('Recognized year: {}'.format(recognized_year))

    return str(datetime.today().year) not in recognized_year
