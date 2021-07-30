import cv2
import logging

from threading import current_thread, Thread

from argus.frame_grabber import FrameGrabber
from argus.frame_saver import FrameSaver
from argus.helpers.telegram import Telegram
from argus.recognizers.openvino import OpenVinoRecognizer


WHITE_COLOR = (255, 255, 255)

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
    logger.info('Thread %s started' % current_thread().name)

    while True:
        alarm = False
        objects_detected = False

        frame = frame_grabber.make_snapshot()

        if save_every_n_frame is not None:
            current_frame_count += 1
            if current_frame_count == save_every_n_frame:
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
            frame_uri = frame_saver.save(frame, prefix='detected')
            if alarm and telegram is not None:
                telegram.send_and_be_silent('Objects detected: %s' % frame_uri)


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
