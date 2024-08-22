import cv2
import logging
import os

from datetime import datetime

from argus.globals import config

WHITE_COLOR = (255, 255, 255)

logger = logging.getLogger('json')


class QueueItem:
    def __init__(self, frame, thread_name):

        self.frame = frame
        self.thread_name = thread_name

        source_config = config['sources'][thread_name]
 
        self.important_objects = source_config.get('important_objects', ['person'])
        self.important_armed_objects = source_config.get('important_armed_objects', ['car'])
        self.detectable_objects = self.important_objects + self.important_armed_objects + source_config.get('other_objects', [])

        self.stills_dir = source_config['stills_dir']
        self.save_every_sec = source_config['save_every_sec']
        self.host_stills_uri = source_config.get('host_stills_uri')

        self.objects_detected = False
        self.important_objects_detected = False
        self.is_armed = False

    def __mark_object(self, obj):
        label = f"{obj['label']} ({obj['confidence']:.2f})"
        label_position = (obj['xmin'], obj['ymin'] - 7)
        cv2.rectangle(self.frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), WHITE_COLOR, 1)
        cv2.putText(self.frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 0.4, WHITE_COLOR, 1)

    def mark_as_recognized(self):
        cv2.putText(self.frame, "Recognized", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    def map_detections_to_frame(self, detection):
        for obj in detection:
            if obj['label'] in self.detectable_objects:
                self.objects_detected = True
                self.__mark_object(obj)
                logger.warning('Object detected', extra=obj)
                if obj['label'] in self.important_objects or (obj['label'] in self.important_armed_objects and self.is_armed):
                    self.important_objects_detected = True
                    self.save_every_sec = 1
                

    def save(self):

        if not os.path.exists(self.stills_dir):
            os.makedirs(self.stills_dir)

        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        if self.objects_detected:
            frame_filename = '{}-{}.jpg'.format(timestamp, 'detected')
        else:
            frame_filename = '{}.jpg'.format(timestamp)

        if not cv2.imwrite(os.path.join(self.stills_dir, frame_filename), self.frame):
            logger.error('Unable to save file: %s' % frame_filename)

        return f'{self.host_stills_uri}/{frame_filename}' if self.host_stills_uri is not None else None
