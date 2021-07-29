import cv2
import logging

from threading import current_thread, Thread

from argus.frame_grabber import FrameGrabber
from argus.frame_saver import FrameSaver
from argus.helpers.telegram import Telegram
from argus.recognizers.openvino import OpenVinoRecognizer


SAVE_EVERY_FRAME = 30

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


def async_run(frame_grabber, frame_saver, recocnizer, telegram):
    
    current_frame_count = 0
    logger.info('Thread %s started' % current_thread().name)
    
    while True:
        alarm = False

        frame = frame_grabber.make_snapshot()

        if current_frame_count == SAVE_EVERY_FRAME:
            frame_saver.save(frame)
            current_frame_count = 0
        else:
            current_frame_count += 1

        objects = recocnizer.split_and_recocnize(frame)

        # Filter objects with correct area and save only DETECTABLE_OBJECTS
        objects = [
            obj for obj in objects
            if obj['total_area'] < MAX_TOTAL_AREA_FOR_OBJECT
            and obj['label'] in DETECTABLE_OBJECTS
        ]

        if objects:
            
            for obj in objects:
                mark_object_on_frame(frame, obj)
                if obj['label'] in IMPORTANT_OBJECTS:
                    alarm = True
                logger.warning(obj)

            file_uri = frame_saver.save(frame, prefix='detected')

            if alarm and telegram is not None:
                telegram.send_and_be_silent('Objects detected: %s ' % file_uri)


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
                FrameSaver(config=config['sources'][source]),
                recocnizer,
                telegram,
            )
        )
        thread.name = source
        thread.start()
