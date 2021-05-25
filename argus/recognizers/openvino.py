import cv2
import logging
import ngraph as ng
import os
import sys

from openvino.inference_engine import IECore
from usb.core import find as finddev

from argus.recognizers import Recognizer
from argus.helpers.yolo import get_objects, filter_objects
from argus.helpers.timing import timing

PROB_THRESHOLD = 0.4

logger = logging.getLogger(__file__)


class OpenVinoRecognizer(Recognizer):

    def init_network(self, config):
        ie = IECore()
        self.net = ie.read_network(
	        os.path.join(config['model_path'], 'frozen_darknet_yolov4_model.xml'), 
	        os.path.join(config['model_path'], 'frozen_darknet_yolov4_model.bin')
	    )

        self.function_from_cnn = ng.function_from_cnn(self.net)

        #Extract network params
        self.input_blob = next(iter(self.net.input_info))
        _, _, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape
        self.exec_net = ie.load_network(network=self.net, device_name=config['device_name'])

        with open(os.path.join(config['model_path'], 'coco.names'), 'r') as f:
            self.labels_map = [x.strip() for x in f]

    @timing
    def recognize(self, frame):
        proc_image = cv2.resize(frame, (self.h, self.w), interpolation=cv2.INTER_LINEAR)
        proc_image = proc_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW

        try:
            result = self.exec_net.infer({self.input_blob: proc_image})
        except:
            logger.exception("Exec Network is down")
            if self.mode == 'production': # Reset usb device. Find ids with lsusb
                dev = finddev(idVendor=0x0424, idProduct=0x9514)
                dev.reset()
            sys.exit(0)


        objects = get_objects(
            result, 
            self.net,
            (self.h, self.w), 
            (frame.shape[0], frame.shape[1]), 
            PROB_THRESHOLD,
            self.function_from_cnn
        )

        objects = filter_objects(objects, iou_threshold=0.4, prob_threshold=PROB_THRESHOLD)

        def _add_object_label_and_total_area(elem):
            elem['object_label'] = self.labels_map[elem['class_id']]
            elem['total_area'] = (elem['ymax'] - elem['ymin']) * (elem['xmax'] - elem['xmin'])
            return elem
        
        objects = list(map(_add_object_label_and_total_area, objects))

        return objects