import logging
import threading

from datetime import datetime, timedelta
from queue import LifoQueue, Empty
from threading import Thread
from time import sleep

from argus.frame_grabber import FrameGrabber
from argus.frame_saver import FrameSaver
from argus.helpers.telegram import Telegram
from argus.recognizers.openvino import OpenVinoRecognizer


SILENT_TIME = timedelta(minutes=30)
SAVE_FRAMES_AFTER_DETECT_OBJECTS = timedelta(seconds=15)

WARNING_QUEUE_SIZE = 10
QUEUE_TIMEOUT = 10
SLEEP_TIME_IF_QUEUE_IS_EMPTY = 5

logger = logging.getLogger('json')

frames = LifoQueue(maxsize=WARNING_QUEUE_SIZE*2)


class QueueElem:

    def __init__(self, frame, thread_name, index_number):
        self.frame = frame
        self.thread_name = thread_name
        self.index_number = index_number


class SnapshotThread(Thread):
    def __init__(self, name, config):
        super(SnapshotThread, self).__init__()
        self.name = name
        self.frames_count = 0
        self.frame_grabber = FrameGrabber(config=config['sources'][name])

    def run(self):
        while True:
            frames.put(QueueElem(
                self.frame_grabber.make_snapshot(),
                self.name,
                self.frames_count
            ))
            self.frames_count += 1


def check_and_restart_dead_snapshot_threads(config):
    active_threads = [t.name for t in threading.enumerate()]
    for thread_name in config['sources']:
        if thread_name not in active_threads:
            thread = SnapshotThread(thread_name, config)
            thread.start()
            logger.warning('Thread %s restarted' % thread_name)


def run(config):

    recocnizer = OpenVinoRecognizer(config)

    silent_notify_until_time = datetime.now()

    if 'telegram_bot' in config:
        telegram = Telegram(config['telegram_bot'])
    else:
        telegram = None

    for source in config['sources']:
        thread = SnapshotThread(source, config)
        thread.start()
        logger.info('Thread %s started' % source)

    last_detection = {}

    frame_saver = FrameSaver(config['sources'])

    while True:

        check_and_restart_dead_snapshot_threads(config)

        try:
            queue_elem = frames.get(timeout=QUEUE_TIMEOUT)
        except Empty:
            logger.error("Queue is empty for %s sec." % QUEUE_TIMEOUT)
            sleep(SLEEP_TIME_IF_QUEUE_IS_EMPTY)
            continue

        queue_size = frames.qsize()
        if queue_size > WARNING_QUEUE_SIZE:
            logger.warning('Warning queue size: %s' % queue_size, extra={'queue_size': queue_size})

        # Save all frames N sec after objects detection or every N frame (defined in config)
        save_every_n_frame = config['sources'][queue_elem.thread_name].get('save_every_n_frame')
        frame_needs_to_be_save = (
            (
                save_every_n_frame and
                queue_elem.index_number % save_every_n_frame == 0

            ) or
            (
                last_detection.get(queue_elem.thread_name) is not None and
                last_detection[queue_elem.thread_name] + SAVE_FRAMES_AFTER_DETECT_OBJECTS > datetime.now()
            )
        )

        if frame_needs_to_be_save:
            frame_saver.save(queue_elem)

        request_id = recocnizer.get_request_id()
        state, processed_queue_elem = recocnizer.get_result(request_id)
        recocnizer.send_to_recocnize(queue_elem, request_id)

        if state.objects_detected:
            last_detection[processed_queue_elem.thread_name] = datetime.now()
            frame_uri = frame_saver.save(processed_queue_elem, prefix='detected')
            if (
                state.alarm and
                telegram is not None and
                last_detection[processed_queue_elem.thread_name] > silent_notify_until_time
            ):
                telegram.send_message('Objects detected: %s' % frame_uri)
                silent_notify_until_time = datetime.now() + SILENT_TIME
