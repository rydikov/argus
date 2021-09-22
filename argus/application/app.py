import logging
import threading

from datetime import datetime, timedelta
from queue import LifoQueue, Empty
from threading import Thread
from time import sleep

from argus.domain.queue_item import QueueItem
from argus.utils.frame_grabber import FrameGrabber
from argus.utils.recognizer import OpenVinoRecognizer
from argus.utils.telegram import Telegram


SILENT_TIME = timedelta(minutes=30)
SAVE_FRAMES_AFTER_DETECT_OBJECTS = timedelta(seconds=15)

WARNING_QUEUE_SIZE = 10
QUEUE_TIMEOUT = 10
SLEEP_TIME_IF_QUEUE_IS_EMPTY = 5

logger = logging.getLogger('json')

frame_items_queue = LifoQueue(maxsize=WARNING_QUEUE_SIZE*2)


class SnapshotThread(Thread):
    def __init__(self, name, config):
        super(SnapshotThread, self).__init__()
        self.name = name
        self.source_config = config['sources'][name]
        self.frames_count = 0
        self.frame_grabber = FrameGrabber(config=self.source_config)

    def run(self):
        while True:
            frame_items_queue.put(QueueItem(
                self.source_config,
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

    recocnizer = OpenVinoRecognizer(config['recognizer'])

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

    while True:

        check_and_restart_dead_snapshot_threads(config)

        try:
            queue_item = frame_items_queue.get(timeout=QUEUE_TIMEOUT)
        except Empty:
            logger.error("Queue is empty for %s sec." % QUEUE_TIMEOUT)
            sleep(SLEEP_TIME_IF_QUEUE_IS_EMPTY)
            continue

        queue_size = frame_items_queue.qsize()
        if queue_size > WARNING_QUEUE_SIZE:
            logger.warning('Warning queue size: %s' % queue_size, extra={'queue_size': queue_size})

        # Save forced all frames N sec after objects detection
        need_save_after_detection = (
            last_detection.get(queue_item.thread_name) is not None and
            last_detection[queue_item.thread_name] + SAVE_FRAMES_AFTER_DETECT_OBJECTS > datetime.now()
        )
        queue_item.save_if_need(forced=need_save_after_detection)

        request_id = recocnizer.get_request_id()
        processed_queue_item = recocnizer.get_result(request_id)
        recocnizer.send_to_recocnize(queue_item, request_id)

        if processed_queue_item is not None and processed_queue_item.objects_detected:
            last_detection[processed_queue_item.thread_name] = datetime.now()
            frame_uri = processed_queue_item.save_if_need(forced=True, prefix='detected')
            if (
                processed_queue_item.important_objects_detected and
                telegram is not None and
                last_detection[processed_queue_item.thread_name] > silent_notify_until_time
            ):
                telegram.send_message('Objects detected: %s' % frame_uri)
                silent_notify_until_time = datetime.now() + SILENT_TIME
