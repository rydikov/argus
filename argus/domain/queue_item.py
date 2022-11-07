import cv2
import logging
import os

from datetime import datetime

WHITE_COLOR = (255, 255, 255)

logger = logging.getLogger('json')


class QueueItem:
    def __init__(self, source_config, frame, thread_name):

        self.frame = frame
        self.thread_name = thread_name
 
        self.important_objects = source_config['important_objects']
        self.detectable_objects = self.important_objects + source_config.get('other_objects', [])
        self.max_object_area = source_config['max_object_area']

        self.stills_dir = source_config['stills_dir']
        self.host_stills_uri = source_config['host_stills_uri']

        self.objects_detected = False
        self.important_objects_detected = False

    def __mark_object(self, obj):
        label = '{}: {} %'.format(obj['label'], round(obj['confidence'] * 100, 1))
        label_position = (obj['xmin'], obj['ymin'] - 7)
        cv2.rectangle(self.frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), WHITE_COLOR, 1)
        cv2.putText(self.frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 0.4, WHITE_COLOR, 1)

    def map_objects_to_frame(self, objects, labels):
        for obj in objects:
            obj['label'] = labels[obj['class_id']]
            obj_area = (obj['ymax'] - obj['ymin']) * (obj['xmax'] - obj['xmin'])
            if obj['label'] in self.detectable_objects and obj_area < self.max_object_area:
                self.objects_detected = True
                self.__mark_object(obj)
                logger.warning('Object detected', extra=obj)
                if obj['label'] in self.important_objects:
                    self.important_objects_detected = True

    def save(self, prefix=None):
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        if not os.path.exists(self.stills_dir):
            os.makedirs(self.stills_dir)

        if prefix is None:
            frame_name = '{}.jpg'.format(timestamp)
        else:
            frame_name = '{}-{}.jpg'.format(timestamp, prefix)

        if not cv2.imwrite(os.path.join(self.stills_dir, frame_name), self.frame):
            logger.error('Unable to save file: %s' % frame_name)

        return '{}/{}'.format(self.host_stills_uri, frame_name)
