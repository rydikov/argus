import cv2
import logging
import os
import sys
import time

from datetime import datetime
from threading import Thread

from argus.frame_grabber import FrameGrabber
from argus.helpers.telegram import Telegram
from argus.recognizers.openvino import OpenVinoRecognizer


SAVE_EVERY_FRAME = 5

WHITE_COLOR = (255, 255, 255)
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
    label = '{}: {} %'.format(obj['label'], round(obj['confidence'] * 100, 1))
    label_position = (obj['xmin'], obj['ymin'] - 7)
    
    cv2.rectangle(
        frame,
        (obj['xmin'], obj['ymin']),
        (obj['xmax'], obj['ymax']),
        WHITE_COLOR,
        1
    )
    cv2.putText(
        frame,
        label,
        label_position,
        cv2.FONT_HERSHEY_COMPLEX,
        0.4,
        WHITE_COLOR,
        1
    )


def save_frame(frame, stills_dir, prefix=None):
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    if not os.path.exists(stills_dir):
        os.makedirs(stills_dir)

    if prefix is None:
        file_name = '{}.jpg'.format(timestamp)
    else:
        file_name = '{}-{}.jpg'.format(timestamp, prefix)

    file_path = os.path.join(stills_dir, file_name)
    is_saved = cv2.imwrite(file_path, frame)
    if not is_saved:
        logger.error('Unable to save file. Prefix: {}'.format(prefix))
    return file_name


def async_run(frame_grabber, recocnizer, telegram, host_stills_uri):
    
    current_frame_count = 0
    
    while True:
        alarm = False

        frame, stills_dir = frame_grabber.make_snapshot()
        if frame is None:
            logger.error("Unable to get frame")
            sys.exit(1)

        current_frame_count += 1
        if current_frame_count == SAVE_EVERY_FRAME:
            save_frame(frame, stills_dir)
            current_frame_count = 0

        objects = recocnizer.split_and_recocnize(frame)
        logger.info(objects)

        # Filter objects with correct area and save only DETECTABLE_OBJECTS
        objects = [
            obj for obj in objects
            if obj['total_area'] < MAX_TOTAL_AREA_FOR_OBJECT
            and obj['label'] in DETECTABLE_OBJECTS
        ]

        if objects:
            
            for obj in objects:
                mark_object_on_frame(frame, obj)
                logger.warning(obj)
                if obj['label'] in IMPORTANT_OBJECTS:
                    alarm = True


            file_path = save_frame(frame, stills_dir, prefix='detected')

            if alarm and telegram is not None:
                telegram.send_and_be_silent(
                    'Objects detected: {}/{}'.format(host_stills_uri, file_path)
                )


def run(config):

    recocnizer = OpenVinoRecognizer(config=config['recognizer'])

    if 'telegram_bot' in config:
        telegram = Telegram(config['telegram_bot'])
    else:
        telegram = None

    for source in config['sources']:
        thread = Thread(
            target=async_run,
            args=(
                FrameGrabber(config=config['sources'][source]),
                recocnizer,
                telegram,
                config['sources'][source]['host_stills_uri'],
            )
        )
        thread.name = source
        thread.start()
