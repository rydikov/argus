import yaml
import time
import sys
import os 
import numpy as np
import logging
import ffmpeg
import cv2
import collections

from datetime import datetime, timedelta
from openvino.inference_engine import IECore

from helpers.yolo import get_objects, filter_objects
from helpers.timing import timing
from helpers.telegram import send_message

# 25 sec
DEADLINE_IN_MSEC = 25000000
PROB_THRESHOLD = 0.4
IMPORTANT_OBJECTS = ['person', 'car', 'cow']

MODE = os.environ.get('MODE', 'development')

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S'
    )

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, 'conf/app.yml')) as f:
    config = yaml.safe_load(f)[MODE]

with open(os.path.join(config['model_path'], 'coco.names'), 'r') as f:
    labels_map = [x.strip() for x in f]

ie = IECore()
net = ie.read_network(
	os.path.join(config['model_path'], 'frozen_darknet_yolov3_model.xml'), 
	os.path.join(config['model_path'], 'frozen_darknet_yolov3_model.bin')
	)

#Extract network params
input_blob = next(iter(net.input_info))
_, _, h, w = net.input_info[input_blob].input_data.shape
exec_net = ie.load_network(network=net, device_name=config['device_name'])

@timing
def make_snapshot():
    snapshot_path = "{}/{}.png".format(config['stills_dir'], datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))

    stream = ffmpeg.input(
        config['gst'],
        rtsp_transport='tcp',
        stimeout=DEADLINE_IN_MSEC
    )
    stream = stream.output(snapshot_path, vframes=1, pix_fmt='rgb24')
    stream.run(capture_stdout=True, capture_stderr=True)

    return snapshot_path

@timing
def recocnize(frame):

    proc_image = cv2.resize(frame, (h, w), interpolation=cv2.INTER_LINEAR)
    proc_image = proc_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    
    result = exec_net.infer({input_blob: proc_image})

    objects = get_objects(
        result, 
        net, 
        (h, w), 
        (frame.shape[0], frame.shape[1]), 
        PROB_THRESHOLD,
        False
    )

    objects = filter_objects(objects, iou_threshold=0.4, prob_threshold=PROB_THRESHOLD)

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

    def is_image_size_less_avg_size(self, image_size):
        avg_image_size = sum(self.last_image_sizes)/self.store_images
        return (image_size/avg_image_size)*100 < self.deviation_percent

    def is_bad(self, image_path):
        image_size = os.path.getsize(image_path)
        self.last_image_sizes.appendleft(image_size)

        if len(self.last_image_sizes) < self.store_images:
            return False
        
        return self.is_image_size_less_avg_size(image_size)


bfc = BadFrameChecker()
last_time_detected = None
silent_to_time = datetime.now()

while True:
    snapshot_delay = 30

    try:
        snapshot_path = make_snapshot()
    except ffmpeg._run.Error as e:
        logging.exception("Time out Error")
        continue

    is_bad_file = bfc.is_bad(snapshot_path)
    if is_bad_file:
        logger.warning('Bad file deleted')
        os.remove(snapshot_path)
        continue

    frame = cv2.imread(snapshot_path)

    objects = recocnize(frame)

    if objects:
        snapshot_delay = 5
        # Draw rectangle
        for obj in objects:
            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255,255,255), 1)
            if obj['object_label'] in IMPORTANT_OBJECTS:
                last_time_detected = datetime.now()
                snapshot_delay = 0
                logger.warning(obj)

        file_name = '{}-detected.png'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        is_saved = cv2.imwrite(os.path.join(config['stills_dir'], file_name), frame)
        if not is_saved:
            logger.error('Unable to save file with detected objects')
            continue

        if last_time_detected and last_time_detected > silent_to_time:
            send_message('Objects detected: {} https://web.rydikov-home.keenetic.pro/Stills/{}'.format(
                ' '.join([obj['object_label'] for obj in objects]),
                file_name
            ))
            silent_to_time = datetime.now() + timedelta(hours=1)
    
    if MODE == 'production':
        time.sleep(snapshot_delay)
