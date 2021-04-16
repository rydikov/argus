import yaml
import time
import sys
import os 
import numpy as np
import logging
import ffmpeg
import cv2
import collections

from usb.core import find as finddev
from datetime import datetime, timedelta
from openvino.inference_engine import IECore

from helpers.yolo import get_objects, filter_objects
from helpers.timing import timing
from helpers.telegram import send_message

# 25 sec
DEADLINE_IN_MSEC = 25000000
PROB_THRESHOLD = 0.4
IMPORTANT_OBJECTS = ['person', 'car', 'cow']
DETECTABLE_OBJECTS = IMPORTANT_OBJECTS + [
    'bicycle', 
    'motorcycle',
    'bird',
    'cat',
    'dog',
    'horse'
    ]

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
def make_rtsp_snapshot(snapshot_path):
    stream = ffmpeg.input(config['source'], rtsp_transport='tcp', stimeout=DEADLINE_IN_MSEC)
    stream = stream.output(snapshot_path, vframes=1, pix_fmt='rgb24')
    try:
        stream.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg._run.Error as e:
        logging.exception("Time out Error")
        raise
    

def make_video_snapshot(snapshot_path):
    __, frame = cap.read()
    is_saved = cv2.imwrite(snapshot_path, frame)
    if not is_saved:
        raise


@timing
def recocnize(frame):

    proc_image = cv2.resize(frame, (h, w), interpolation=cv2.INTER_LINEAR)
    proc_image = proc_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW

    try:
        result = exec_net.infer({input_blob: proc_image})
    except:
        logging.exception("Exec Network is down")
        if MODE == 'production': # Reset usb device. Find ids with lsusb
            dev = finddev(idVendor=0x0424, idProduct=0x9514)
            dev.reset()
        sys.exit(0)


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


def split_and_recocnize(frame):

    h, w = frame.shape[0], frame.shape[1] # e.g. 1080x1920
    half_frame = int(w/2) # 960
    left_frame = frame[h-half_frame:h, 0:half_frame] # [120:1080, 0:960]
    right_frame = frame[h-half_frame:h, half_frame:w] # [120:1080, 960:1920]

    left_frame_objects = recocnize(left_frame)
    right_frame_objects = recocnize(right_frame)

    for obj in left_frame_objects:
        obj['ymin'] += h - half_frame
        obj['ymax'] += h - half_frame

    for obj in right_frame_objects:
        obj['xmin'] += half_frame
        obj['ymin'] += h - half_frame
        obj['xmax'] += half_frame
        obj['ymax'] += h - half_frame

    return left_frame_objects + right_frame_objects


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

if config['source'].startswith('rtsp'):
    snapshot_method = make_rtsp_snapshot
else:
    cap = cv2.VideoCapture(config['source'])
    snapshot_method = make_video_snapshot

while True:
    snapshot_delay = 5
    snapshot_path = "{}/{}.png".format(config['stills_dir'], datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))

    try:
        snapshot_method(snapshot_path)
    except:
        logging.exception('Unable to get snapshot')
        continue

    if bfc.is_bad(snapshot_path):
        logger.warning('Bad file deleted')
        os.remove(snapshot_path)
        continue

    frame = cv2.imread(snapshot_path)
    objects = split_and_recocnize(frame)

    has_detectable_objects = set([obj['object_label'] for obj in objects]) & set(DETECTABLE_OBJECTS)

    if has_detectable_objects:
        snapshot_delay = 0
        # Draw rectangle
        for obj in objects:
            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255,255,255), 1)
            cv2.putText(frame, '{}: {} %'.format(obj['object_label'], round(obj['confidence'] * 100, 1)),
                        (obj['xmin'], obj['ymin'] - 7),
                        cv2.FONT_HERSHEY_COMPLEX, 
                        0.4, 
                        (255,255,255), 
                        1)
            if obj['object_label'] in IMPORTANT_OBJECTS:
                last_time_detected = datetime.now()
                snapshot_delay = 0
                logger.warning(obj)

        file_name = '{}-detected.png'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        is_saved = cv2.imwrite(os.path.join(config['stills_dir'], file_name), frame)
        if not is_saved:
            logger.error('Unable to save file with detected objects')
            continue

        if MODE == 'production' and last_time_detected and last_time_detected > silent_to_time:
            send_message('Objects detected: {} https://web.rydikov-home.keenetic.pro/Stills/{}'.format(
                ' '.join([obj['object_label'] for obj in objects]),
                file_name
            ))
            silent_to_time = datetime.now() + timedelta(hours=1)
    
    if MODE == 'production':
        time.sleep(snapshot_delay)
