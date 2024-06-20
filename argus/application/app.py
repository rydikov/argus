import asyncio
import copy 
import logging
import os
import threading
import sys

from datetime import datetime, timedelta
from collections import deque
from threading import Thread
from time import sleep

from argus.domain.queue_item import QueueItem
from argus.utils.frame_grabber import FrameGrabber
from argus.settings import (
    detected_frame_notification_time, 
    send_frames_after_signal, 
)

from argus.globals import (
    config, 
    recognizer, 
    telegram_service, 
    alarm_system_service,
    aqara_service
)

REDUCE_CPU_USAGE_SEC = 0.01
LOG_TEMPERATURE_TIME = timedelta(minutes=1)
LOG_SOURCE_FPS_TIME = timedelta(minutes=1)

QUEUE_SIZE = 3
DEFAULT_SERVER_RESPONSE = 'OK'

logger = logging.getLogger('json')

# Dict of frame Queues
frame_items_queues = {}

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
            detected_frame_notification_time.clear()
        elif message == 'get_photos':
            send_frames_after_signal.extend(list(frame_items_queues.keys()))
        elif message == 'arming':
            alarm_system_service.arming()
        elif message == 'disarming':
            alarm_system_service.disarming()
        elif message ==  'status':
            telegram_service.send_message(alarm_system_service.status)
        elif message == 'run_scene':
            aqara_service.run_scene()
        elif message.startswith('save_code'):
            _, code = message.split(':')
            aqara_service.save_code(code)

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
    def __init__(self, name):
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
            frame_items_queues[self.name] = deque(maxlen=QUEUE_SIZE)

        while True:

            frame_items_queues[self.name].append(QueueItem(
                self.source_config,
                self.frame_grabber.make_snapshot(),
                self.name)
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


def check_and_restart_dead_snapshot_threads():
    active_threads = [t.name for t in threading.enumerate()]
    
    if 'MainThread' not in active_threads:
        logger.error('MainThread is not running')
        sys.exit(1)

    for thread_name in config['sources']:
        if thread_name not in active_threads:
            thread = SnapshotThread(thread_name)
            thread.start()
            logger.warning('Thread %s restarted' % thread_name)


def run():

    last_log_temperature_time = datetime.now()

    # Create and start threading for every source
    for source in config['sources']:
        thread = SnapshotThread(source)
        thread.start()
        logger.info('Thread %s started' % source)


    # Create and start threading for external events
    ExternalSignalsReciver('127.0.0.1', 8888).start()

    while True:

        sleep(REDUCE_CPU_USAGE_SEC)

        for frame_items_queue in frame_items_queues.values():
            try:
                queue_item = copy.deepcopy(frame_items_queue.pop())
            except IndexError:
                check_and_restart_dead_snapshot_threads()
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

            # Send frame from buffer to recognize
            recognizer.send_to_recognize(queue_item)
            