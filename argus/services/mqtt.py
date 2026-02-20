import logging
import os

import asyncio
import paho.mqtt.publish as publish

logger = logging.getLogger('json')


class MQTTService:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port

    def publish(self, topic, payload):
        publish.single(
            topic=topic,
            payload=payload,
            client_id='ARGUS',
            hostname=self.hostname,
            port=self.port,
        )

    async def publish_async(self, topic, payload):
        await asyncio.to_thread(self.publish, topic, payload)
