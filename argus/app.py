import cv2
import logging
import os 
import time

from datetime import datetime, timedelta
from openvino.inference_engine import IECore
from usb.core import find as finddev

from argus.frame_grabber import FrameGrabber
from argus.helpers.telegram import Telegram
from argus.recognizers.openvino import OpenVinoRecognizer


# Максимальная площадь возможного объекта
WHITE_COLOR = (255,255,255)
MAX_TOTAL_AREA_FOR_OBJECT = 15000
IMPORTANT_OBJECTS = ['person', 'car', 'cow']
DETECTABLE_OBJECTS = IMPORTANT_OBJECTS + [
    'bicycle', 
    'motorcycle',
    'bird',
    'cat',
    'dog',
    'horse'
    ]


logger = logging.getLogger(__file__)


def mark_object_on_frame(frame, obj):
    label = '{}: {} %'.format(obj['object_label'], round(obj['confidence'] * 100, 1))
    label_position = (obj['xmin'], obj['ymin'] - 7)
    cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), WHITE_COLOR, 1)
    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 0.4, WHITE_COLOR, 1)

def run(config, mode):

    last_time_detected = None
    silent_until_time = datetime.now()

    recocnizer = OpenVinoRecognizer(config=config, mode=mode)
    frame_grabber = FrameGrabber(config=config)

    if mode == 'production':
        telegram = Telegram(config)


    while True:
        alarm = False
        
        frame = frame_grabber.make_snapshot()
        if frame is None:
            logger.error("Unable to get frame")
            continue
        
        objects = recocnizer.split_and_recocnize(frame)
        logger.info(object)

        # Filter objects with correct area and save only DETECTABLE_OBJECTS
        objects = [
            obj for obj in objects 
            if obj['total_area'] < MAX_TOTAL_AREA_FOR_OBJECT and obj['object_label'] in DETECTABLE_OBJECTS
            ]

        if objects:
            alarm = True
            # Draw rectangle with text
            for obj in objects:
                mark_object_on_frame(frame, obj)
                logger.warning(obj)

            file_name = '{}-detected.jpg'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            is_saved = cv2.imwrite(os.path.join(config['stills_dir'], file_name), frame)
            if not is_saved:
                logger.error('Unable to save file with detected objects')
                continue

            if mode == 'production':
                telegram.send_and_be_silent('Objects detected: {}/{}'.format(config['host_stills_dir'], file_name))
        
        if mode == 'production':
            time.sleep(0 if alarm else 5)
