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
SAVE_FRAMES_AFTER_DETECT_OBJECTS = timedelta(minutes=1)

WARNING_QUEUE_SIZE = 10
QUEUE_TIMEOUT = 10
SLEEP_TIME_IF_QUEUE_IS_EMPTY = 5

logger = logging.getLogger('json')

frames = LifoQueue(maxsize=WARNING_QUEUE_SIZE*2)
snapshot_threads_names = []


class SnapshotThread(Thread):
    def __init__(self, name, config):
        super(SnapshotThread, self).__init__()
        self.name = name
        self.frame_grabber = FrameGrabber(config=config['sources'][name])

    def run(self):
        while True:
            frames.put({
                'frame': self.frame_grabber.make_snapshot(),
                'thread_name': self.name
            })


def check_and_restart_dead_snapshot_threads(config):
    active_threads = [t.name for t in threading.enumerate()]
    for thread_name in snapshot_threads_names:
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

    frame_savers = {}

    for source in config['sources']:
        thread = SnapshotThread(source, config)
        thread.start()
        logger.info('Thread %s started' % source)
        snapshot_threads_names.append(thread.name)
        frame_savers[thread.name] = FrameSaver(config=config['sources'][source])

    time_of_last_object_detection = {}

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

        source_frame = queue_elem['frame']
        source_thread_name = queue_elem['thread_name']

        # Safe all frames forced N sec after objects detect
        forced_save = (
            time_of_last_object_detection.get(source_thread_name) is not None and
            time_of_last_object_detection[source_thread_name] + SAVE_FRAMES_AFTER_DETECT_OBJECTS > datetime.now()
        )

        frame_savers[source_thread_name].save_if_need(source_frame, forced_save)

        request_id = recocnizer.get_request_id()
        status, processed_frame, processed_thread_name = recocnizer.get_result(request_id)
        recocnizer.send_to_recocnize(source_frame, source_thread_name, request_id)

        if status.objects_detected:
            time_of_last_object_detection[processed_thread_name] = datetime.now()
            frame_uri = frame_savers[processed_thread_name].save_if_need(
                processed_frame,
                forced=True,
                prefix='detected'
            )
            if (
                status.alarm and
                telegram is not None and
                time_of_last_object_detection[processed_thread_name] > silent_notify_until_time
            ):
                telegram.send_message('Objects detected: %s' % frame_uri)
                silent_notify_until_time = datetime.now() + SILENT_TIME
