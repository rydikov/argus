import time
import sys
import os 
import numpy as np
import datetime
import cv2
import yaml

from openvino.inference_engine import IECore

from helpers.yolo import get_objects, filter_objects

MODE = 'development'

with open("conf/app.yml") as f:
    config = yaml.safe_load(f)[MODE]

cap = cv2.VideoCapture(config['gst'])
ie = IECore()

prob_threshold = 0.4
iou_threshold = 0.4

net = ie.read_network(
	'../frozen_darknet_yolov4_model.xml', 
	'../frozen_darknet_yolov4_model.bin'
	)

#Extract network params
input_blob = next(iter(net.input_info))
_, _, h, w = net.input_info[input_blob].input_data.shape

exec_net = ie.load_network(network=net, device_name='CPU')

with open("../coco.names", 'r') as f:
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
    
    objects = map(_add_object_label, objects)

    return objects


while True:
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            objects = recocnize(frame)

            # Drow rectangle
            for obj in objects:
                if obj['object_label'] == 'person':
                    print(obj)
                cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255,255,255), 1)

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
                raise
            # time.sleep(snapshot_delay)
    else:
        print("Cap is not avalible")
        sys.exit(0)
