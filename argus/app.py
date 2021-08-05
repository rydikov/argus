import cv2
import logging

from threading import current_thread, Thread
from datetime import datetime, timedelta

from argus.frame_grabber import FrameGrabber
from argus.frame_saver import FrameSaver
from argus.helpers.telegram import Telegram
from argus.recognizers.openvino import OpenVinoRecognizer


WHITE_COLOR = (255, 255, 255)
SILENT_TIME = timedelta(minutes=30)
SAVE_FRAMES_AFTER_DETECT_OBJECTS = timedelta(minutes=3)

logger = logging.getLogger(__file__)


def mark_object_on_frame(frame, obj):
    label = '{}: {} %'.format(obj['label'], round(obj['confidence'] * 100, 1))
    label_position = (obj['xmin'], obj['ymin'] - 7)
    cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), WHITE_COLOR, 1)
    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 0.4, WHITE_COLOR, 1)


def async_run(
    frame_grabber,
    frame_saver,
    recocnizer,
    telegram,
    important_objects,
    detectable_objects,
    max_total_area_for_object,
    save_every_n_frame
):

    current_frame_count = 0
    time_of_the_last_attempt = None
    silent_notify_until_time = datetime.now()

    logger.info('Thread %s started' % current_thread().name)

    while True:
        alarm = False
        objects_detected = False

        frame = frame_grabber.make_snapshot()

        if save_every_n_frame is not None:
            current_frame_count += 1

        if (
            current_frame_count == save_every_n_frame or (
                time_of_the_last_attempt is not None and
                time_of_the_last_attempt + SAVE_FRAMES_AFTER_DETECT_OBJECTS > datetime.now()
            )
            
        ):
            frame_saver.save(frame)
            current_frame_count = 0

        objects = recocnizer.split_and_recocnize(frame)

        for obj in objects:
            # Mark and save objects with correct area
            # and save frame with detectable objects only
            if (
                obj['label'] in detectable_objects and
                (
                    max_total_area_for_object is None or
                    obj['total_area'] < max_total_area_for_object

                )
            ):
                objects_detected = True
                mark_object_on_frame(frame, obj)
                logger.warning(obj)
                if obj['label'] in important_objects:
                    alarm = True

        if objects_detected:
            time_of_the_last_attempt = datetime.now()
            frame_uri = frame_saver.save(frame, prefix='detected')
            if (
                alarm and
                time_of_the_last_attempt > silent_notify_until_time and
                telegram is not None
            ):
                telegram.send_message('Objects detected: %s' % frame_uri)
                silent_notify_until_time = datetime.now() + SILENT_TIME


def run(config):

    recocnizer = OpenVinoRecognizer(config=config['recognizer'])

    if 'telegram_bot' in config:
        telegram = Telegram(config['telegram_bot'])
    else:
        telegram = None

    important_objects = config['important_objects']
    detectable_objects = important_objects + config.get('other_objects', [])

    for source in config['sources']:
        thread = Thread(
            target=async_run,
            args=(
                FrameGrabber(config=config['sources'][source]),
                FrameSaver(config=config['sources'][source]),
                recocnizer,
                telegram,
                important_objects,
                detectable_objects,
                config['sources'][source].get('max_total_area_for_object'),
                config['sources'][source].get('save_every_n_frame')
            )
        )
        thread.name = source
        thread.start()
