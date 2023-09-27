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
REDUCE_CPU_USAGE_SEC = 0.01
LOG_TEMPERATURE_TIME = timedelta(minutes=1)
LOG_RECOGNIZE_RPS_TIME = timedelta(minutes=1)
LOG_SOURCE_FPS_TIME = timedelta(minutes=1)

QUEUE_SIZE = 3
DEFAULT_SERVER_RESPONSE = 'OK'

logger = logging.getLogger('json')

# Dict of frame Queues
frame_items_queues = {}

# Last time when frame has been saved
last_frame_save_time = {}

# No telegram alerting on silent time after detection
silent_notify_until_time = {}

# List for frames to be sent after an external signal
send_frames_after_signal = []

# Recognize RPS
recognize_rps = {'count': 0, 'time': datetime.now()}

# Source FPS
source_fps = {}


class ServerProtocol(asyncio.Protocol):
    def connection_made(self, transport):
        logger.info('Connection from {}'.format(transport.get_extra_info('peername')))
        self.transport = transport

    def data_received(self, data):
        message = data.decode()
        logger.info(f'Data received: {message}')

        if message == 'restart':
            self.transport.write(DEFAULT_SERVER_RESPONSE.encode())
            self.transport.close()
            os._exit(0)
        elif message == 'reset':
            silent_notify_until_time.clear()
        elif message == 'get_photos':
            send_frames_after_signal.extend(list(frame_items_queues.keys()))

        self.transport.write(DEFAULT_SERVER_RESPONSE.encode())

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

            # Remove old item from queue if it is full
            if frame_items_queues[self.name].full():
                frame_items_queues[self.name].get()

            frame_items_queues[self.name].put(QueueItem(
                self.source_config,
                self.frame_grabber.make_snapshot(),
                self.name,
            ),
            block=False
            )

            # Log FPS to logs
            if self.name in source_fps:
                source_fps[self.name]['count'] += 1
                now = datetime.now()
                if source_fps[self.name]['time'] + LOG_SOURCE_FPS_TIME < now:
                    delta = now - source_fps[self.name]['time']
                    fps = source_fps[self.name]['count'] / delta.seconds
                    source_fps[self.name]['time'] = now
                    source_fps[self.name]['count'] = 0
                    logger.info(f'FPS from {self.name}: {fps}', extra={'fps': fps, 'source': self.name})
            else:
                source_fps[self.name] = {'count': 0, 'time': datetime.now()}


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

        # Write RPS to logs
        recognize_rps['count'] += 1
        now = datetime.now()
        if recognize_rps['time'] + LOG_RECOGNIZE_RPS_TIME < now:
            delta = now - recognize_rps['time']
            rps = recognize_rps['count'] / delta.seconds
            recognize_rps['time'] = now
            recognize_rps['count'] = 0
            logger.info(f'Recognized RPS: {rps}', extra={'rps': rps})

    return processed_queue_item


def run(config):

    recognizer = OpenVinoRecognizer(config['recognizer'])

    last_detection = {}
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

            # Log temperature every LOG_TEMPERATURE_TIME
            if last_log_temperature_time + LOG_TEMPERATURE_TIME < datetime.now():
                try:
                    recognizer.log_temperature()
                except RuntimeError as e:
                    logger.warning(f'Unable to get device temperature {e}')
                    os._exit(0)
                else:
                    last_log_temperature_time = datetime.now()

            # Get recognized frame and send frame from buffer to recognize
            try:
                processed_queue_item = get_and_set(recognizer, queue_item)
            except RuntimeError as e:
                logger.warning(f'An exception occurred in the main thread {e}')
                os._exit(0)

            if processed_queue_item is None:
                continue
            else:
                thread_name = processed_queue_item.thread_name

            if processed_queue_item.objects_detected:
                
                previous_last_detection = last_detection.get(thread_name)
                last_detection[thread_name] = datetime.now()

                # Save detected frames no more than once per second
                if (
                    previous_last_detection is not None 
                    and last_detection[thread_name] - previous_last_detection > timedelta(seconds=1)
                ):
                    frame_uri = processed_queue_item.save(prefix='detected')
                    # Telegram alerting
                    if (
                        processed_queue_item.important_objects_detected and
                        telegram is not None and
                        last_detection[thread_name] > silent_notify_until_time.get(thread_name, datetime.now() - SILENT_TIME)
                    ):
                        telegram.send_message(f'Objects detected: {frame_uri}')
                        silent_notify_until_time[thread_name] = datetime.now() + SILENT_TIME

            else:

                # Save frame every N sec
                delta = timedelta(seconds=config['sources'][thread_name]['save_every_sec'])
                
                if thread_name not in last_frame_save_time:
                    last_frame_save_time[thread_name] = datetime.now() - delta

                if last_frame_save_time[thread_name] + delta < datetime.now():
                    processed_queue_item.save()
                    last_frame_save_time[thread_name] = datetime.now()


            if thread_name in send_frames_after_signal and telegram is not None:
                send_frames_after_signal.remove(thread_name)
                telegram.send_frame(processed_queue_item.frame, f'Photo from {thread_name}')

