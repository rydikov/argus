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


class SnapshotThread(Thread):
    def __init__(self, name, config):
        super(SnapshotThread, self).__init__()
        self.name = name
        self.frame_grabber = FrameGrabber(config=config['sources'][name])
        self.frames_count = 0

    def run(self):
        while True:
            frames.put({
                'frame': self.frame_grabber.make_snapshot(),
                'thread_name': self.name,
                'index_number': self.frames_count
            })
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

        # Save all frames N sec after objects detection or every N frame (defined in config)
        save_every_n_frame = config['sources'][source_thread_name].get('save_every_n_frame')
        frame_needs_to_be_save = (
            (
                save_every_n_frame and
                queue_elem['index_number'] % save_every_n_frame == 0

            ) or
            (
                time_of_last_object_detection.get(source_thread_name) is not None and
                time_of_last_object_detection[source_thread_name] + SAVE_FRAMES_AFTER_DETECT_OBJECTS > datetime.now()
            )
        )

        if frame_needs_to_be_save:
            FrameSaver.save(source_frame, config['sources'][source_thread_name]['stills_dir'])

        request_id = recocnizer.get_request_id()
        status, processed_frame, processed_thread_name = recocnizer.get_result(request_id)
        recocnizer.send_to_recocnize(source_frame, source_thread_name, request_id)

        if status.objects_detected:
            time_of_last_object_detection[processed_thread_name] = datetime.now()
            frame_name = FrameSaver.save(
                processed_frame,
                config['sources'][processed_thread_name]['stills_dir'],
                prefix='detected'
            )
            if (
                status.alarm and
                telegram is not None and
                time_of_last_object_detection[processed_thread_name] > silent_notify_until_time
            ):
                frame_uri = '{}/{}'.format(config['sources'][processed_thread_name]['host_stills_uri'], frame_name)
                telegram.send_message('Objects detected: %s' % frame_uri)
                silent_notify_until_time = datetime.now() + SILENT_TIME
