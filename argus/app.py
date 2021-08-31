import queue
import cv2
import logging

from threading import current_thread, Thread
from queue import LifoQueue
from datetime import datetime, timedelta

from argus.frame_grabber import FrameGrabber
from argus.frame_saver import FrameSaver
from argus.helpers.telegram import Telegram
from argus.recognizers.openvino import OpenVinoRecognizer


WHITE_COLOR = (255, 255, 255)
SILENT_TIME = timedelta(minutes=30)
SAVE_FRAMES_AFTER_DETECT_OBJECTS = timedelta(minutes=1)

logger = logging.getLogger('json')

WARNING_QUEUE_SIZE = 10
frames = LifoQueue(maxsize=WARNING_QUEUE_SIZE*2)


def mark_object_on_frame(frame, obj):
    label = '{}: {} %'.format(obj['label'], round(obj['confidence'] * 100, 1))
    label_position = (obj['xmin'], obj['ymin'] - 7)
    cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), WHITE_COLOR, 1)
    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 0.4, WHITE_COLOR, 1)


def async_snapshot(frame_grabber):
    thread_name = current_thread().name
    logger.info('Thread %s started' % thread_name)
    while True:
        frames.put({'frame': frame_grabber.make_snapshot(), 'thread_name': thread_name})


def run(config):

    recocnizer = OpenVinoRecognizer(
        config=config['recognizer'], 
        threads_count=len(config['sources'])
    )

    silent_notify_until_time = datetime.now()
    if 'telegram_bot' in config:
        telegram = Telegram(config['telegram_bot'])
    else:
        telegram = None


    frame_savers = {}

    for source in config['sources']:
        thread = Thread(
            target=async_snapshot,
            args=(
                FrameGrabber(config=config['sources'][source]),
            )
        )
        thread.name = source
        thread.start()
        frame_savers[source] = FrameSaver(config=config['sources'][source])

    important_objects = config['important_objects']
    detectable_objects = important_objects + config.get('other_objects', [])
    
    detections_time = {}
    
    while True:
        
        alarm = False
        objects_detected = False
        forced_save = False

        queue_elem = frames.get()

        queue_size = frames.qsize()
        if queue_size > WARNING_QUEUE_SIZE:
            logger.warning('Queue size: %s' % queue_size, extra={'queue_size': queue_size})

        source_frame = queue_elem['frame']
        thread_name = queue_elem['thread_name']

        frame_saver = frame_savers[thread_name]

        if (
            detections_time.get(thread_name) is not None and
            detections_time[thread_name] + SAVE_FRAMES_AFTER_DETECT_OBJECTS > datetime.now()
        ):
            forced_save = True
            
        frame_saver.save_if_need(source_frame, forced_save)
            
        request_id = recocnizer.get_request_id()
        objects, rec_frame, rec_thread_name = recocnizer.get_result(request_id)
        recocnizer.send_to_recocnize(source_frame, thread_name, request_id)

        for obj in objects:
            # Mark and save objects with correct area
            # and save frame with detectable objects only
            if obj['label'] in detectable_objects:
                objects_detected = True
                mark_object_on_frame(rec_frame, obj)
                logger.warning('Object detected', extra=obj)
                if obj['label'] in important_objects:
                    alarm = True

        if objects_detected:
            detections_time[rec_thread_name] = datetime.now()
            frame_saver = frame_savers[rec_thread_name]
            frame_uri = frame_saver.save_if_need(rec_frame, forced=True, prefix='detected')
            if (
                alarm and
                telegram is not None and
                detections_time[rec_thread_name] > silent_notify_until_time
            ):
                telegram.send_message('Objects detected: %s' % frame_uri)
                silent_notify_until_time = datetime.now() + SILENT_TIME
