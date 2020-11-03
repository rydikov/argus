import yaml
import time
import sys
import os 
import numpy as np
import logging
import datetime
import cv2
import collections

from openvino.inference_engine import IECore

from helpers.yolo import get_objects, filter_objects

MODE = os.environ.get('MODE', 'development')

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S'
    )

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, 'conf/app.yml')) as f:
    config = yaml.safe_load(f)[MODE]

cap = cv2.VideoCapture(config['gst'])
ie = IECore()

prob_threshold = 0.4
iou_threshold = 0.4

net = ie.read_network(
	os.path.join(config['model_path'], 'frozen_darknet_yolov4_model.xml'), 
	os.path.join(config['model_path'], 'frozen_darknet_yolov4_model.bin')
	)

#Extract network params
input_blob = next(iter(net.input_info))

_, _, h, w = net.input_info[input_blob].input_data.shape

exec_net = ie.load_network(network=net, device_name=config['device_name'])

with open(os.path.join(config['model_path'], 'coco.names'), 'r') as f:
    labels_map = [x.strip() for x in f]


def recocnize(frame):

    proc_image = cv2.resize(frame, (h, w), interpolation=cv2.INTER_LINEAR)
    proc_image = proc_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    
    result = exec_net.infer({input_blob: proc_image})

    objects = get_objects(
        result, 
        net, 
        (h, w), 
        (frame.shape[0], frame.shape[1]), 
        prob_threshold,
        False
    )

    objects = filter_objects(objects, iou_threshold, prob_threshold)

    def _add_object_label(elem):
        elem['object_label'] = labels_map[elem['class_id']]
        return elem
    
    objects = list(map(_add_object_label, objects))

    return objects


class BadFrameChecker(object):
    
    def __init__(self):
        self.store_images = 5
        self.last_image_sizes = collections.deque([], self.store_images)
        self.deviation_percent = 60

    def is_bad(self, image_path):
        image_size = os.path.getsize(image_path)

        if len(self.last_image_sizes) < self.store_images:
            self.last_image_sizes.appendleft(image_size)
            return False
        else:
            avg_image_size = sum(self.last_image_sizes)/self.store_images
            if (100-image_size/avg_image_size) < self.deviation_percent:
                return True
            else:
                self.last_image_sizes.appendleft(image_size)
        return False


bfc = BadFrameChecker()

while True:
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:

            objects = recocnize(frame)

            # Draw rectangle
            for obj in objects:
                cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255,255,255), 1)
                if obj['object_label'] == 'person':
                    logger.warn(obj)
                else:
                    logging.info(obj)
                

            # Save image
            if objects:
                filename_template = "{}/{}_detected.jpg"
                snapshot_delay = 5
            else:
                filename_template = "{}/{}.jpg"
                snapshot_delay = 30

            path = filename_template.format(config['stills_dir'], datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            is_saved = cv2.imwrite(path, frame)
            if not is_saved:
                logger.error('Unable to save file')
            
            is_bad = bfc.is_bad(path)
            if is_bad:
                logger.warn('Bad file deleted')
                os.remove(path)
            
            if MODE == 'production':
                time.sleep(snapshot_delay)
    else:
        logger.error('Cap is not avalible')
        sys.exit(0)
