import asyncio
import copy 
import logging
import os
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
REDUCE_CPU_USAGE_SEC = 0.01
LOG_TEMPERATURE_TIME = timedelta(minutes=1)

QUEUE_SIZE = 3

logger = logging.getLogger('json')

frame_items_queues = {}
last_frame_save_time = {}
external_alarm_time = {'time': None}



class ServerProtocol(asyncio.Protocol):
    def connection_made(self, transport):
        logger.info('Connection from {}'.format(transport.get_extra_info('peername')))
        self.transport = transport

    def data_received(self, data):
        message = data.decode()
        logger.info('Data received: {!r}'.format(message))

        external_alarm_time['time'] = datetime.now()

        logger.info('Send: {!r}'.format(message))
        self.transport.write(data)

        logger.info('Close the client socket')
        self.transport.close()


class ExternalSignalsReciver(Thread):

    def __init__(self, host, port):
        super(ExternalSignalsReciver, self).__init__()
        self.name = "ExternalSignalsReciver"
        self.host = host
        self.port = port

    async def serve(self):
        loop = asyncio.get_running_loop()
        server = await loop.create_server(
            lambda: ServerProtocol(),
            self.host,
            self.port
        )
        async with server:
            await server.serve_forever()

    def run(self):
        asyncio.run(self.serve())


class SnapshotThread(Thread):
    def __init__(self, name, config):
        super(SnapshotThread, self).__init__()
        self.name = name
        self.source_config = config['sources'][name]
        self.frame_grabber = FrameGrabber(config=self.source_config)

    def run(self):
        """
        Putting QueueItem to the queue in an infinite queue.
        Queue is global. Thread is unique for every source.
        """
        if self.name not in frame_items_queues:
            frame_items_queues[self.name] = LifoQueue(maxsize=QUEUE_SIZE)

        while True:

            if frame_items_queues[self.name].full():
                frame_items_queues[self.name].get()

            frame_items_queues[self.name].put(QueueItem(
                self.source_config,
                self.frame_grabber.make_snapshot(),
                self.name,
            ),
            block=False
            )


def check_and_restart_dead_snapshot_threads(config):
    active_threads = [t.name for t in threading.enumerate()]
    for thread_name in config['sources']:
        if thread_name not in active_threads:
            thread = SnapshotThread(thread_name, config)
            thread.start()
            logger.warning('Thread %s restarted' % thread_name)


def get_and_set(recocnizer, queue_item):
    request_id = recocnizer.get_request_id()
    if request_id is None:
        return None
    processed_queue_item = recocnizer.get_result(request_id)
    recocnizer.send_to_recocnize(queue_item, request_id)
    if processed_queue_item is not None:
        logger.info(
            'Recognized frame getted from {} and frame from {} sended. RID: {}'.format(
                processed_queue_item.thread_name, 
                queue_item.thread_name,
                request_id
            ), 
            extra={'thread_name': queue_item.thread_name}
        )
    return processed_queue_item


def run(config):

    recognizer = OpenVinoRecognizer(config['recognizer'])

    silent_notify_until_time = datetime.now()
    last_log_temperature_time = datetime.now()

    if 'telegram_bot' in config:
        telegram = Telegram(config['telegram_bot'])
    else:
        telegram = None

    # Create and start threading for every source
    for source in config['sources']:
        thread = SnapshotThread(source, config)
        thread.start()
        logger.info('Thread %s started' % source)

    # Dict with last time detecton for every source
    last_detection = {}

    # Create and start threading for external events
    ExternalSignalsReciver('localhost', 8888).start()

    while True:

        sleep(REDUCE_CPU_USAGE_SEC)

        check_and_restart_dead_snapshot_threads(config)

        for frame_items_queue in frame_items_queues.values():
            try:
                queue_item = copy.deepcopy(frame_items_queue.get(block=False))
            except Empty:
                continue

            # Save forced all frames N sec after objects detection
            need_save_after_detection = (
                last_detection.get(queue_item.thread_name) is not None and
                last_detection[queue_item.thread_name] + SAVE_FRAMES_AFTER_DETECT_OBJECTS > datetime.now()
            )
            #FIXME: Move to params
            need_save_after_detection = False

            # Save forced all frames N sec after recived external signal
            need_save_after_external_signal = (
                external_alarm_time['time'] is not None and
                external_alarm_time['time'] + SAVE_FRAMES_AFTER_DETECT_OBJECTS > datetime.now()
            )

            # Save frame every N sec
            delta = timedelta(seconds=config['sources'][queue_item.thread_name]['save_every_sec'])
            
            if queue_item.thread_name not in last_frame_save_time:
                last_frame_save_time[queue_item.thread_name] = datetime.now() - delta

            if last_frame_save_time[queue_item.thread_name] + delta < datetime.now():
                need_save_save_by_time = True
                last_frame_save_time[queue_item.thread_name] = datetime.now()
            else:
                need_save_save_by_time = False

            if any([need_save_after_detection, need_save_after_external_signal, need_save_save_by_time]):
                queue_item.save()

            # Log temperature every LOG_TEMPERATURE_TIME
            if last_log_temperature_time + LOG_TEMPERATURE_TIME < datetime.now():
                try:
                    recognizer.log_temperature()
                except RuntimeError as e:
                    logger.warning(f'Unable to get device temperature {e}')
                else:
                    last_log_temperature_time = datetime.now()

            try:
                processed_queue_item = get_and_set(recognizer, queue_item)
            except RuntimeError as e:
                logger.warning(f'An exception occurred in the main thread {e}')
                os._exit(0)
            
            if processed_queue_item is not None and processed_queue_item.objects_detected:
                last_detection[processed_queue_item.thread_name] = datetime.now()
                frame_uri = processed_queue_item.save(prefix='detected')

                # Telegram alerting
                if (
                    processed_queue_item.important_objects_detected and
                    telegram is not None and
                    last_detection[processed_queue_item.thread_name] > silent_notify_until_time
                ):
                    telegram.send_message('Objects detected: %s' % frame_uri)
                    silent_notify_until_time = datetime.now() + SILENT_TIME
