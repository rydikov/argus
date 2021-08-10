import cv2
import logging
import ngraph as ng
import os
import sys

from openvino.inference_engine import IECore
from usb.core import find as finddev

from argus.helpers.timing import timing
from argus.recognizers import Recognizer
from argus.helpers.yolo import get_objects, filter_objects

PROB_THRESHOLD = 0.4

logger = logging.getLogger('json')


class OpenVinoRecognizer(Recognizer):

    def init_network(self):
        ie = IECore()
        self.net = ie.read_network(
            os.path.join(
                self.config['model_path'],
                'frozen_darknet_yolov4_model.xml'
            ),
            os.path.join(
                self.config['model_path'],
                'frozen_darknet_yolov4_model.bin'
            )
        )

        self.function_from_cnn = ng.function_from_cnn(self.net)

        # Extract network params
        self.input_blob = next(iter(self.net.input_info))

        _, _, self.h, self.w = \
            self.net.input_info[self.input_blob].input_data.shape

        self.exec_net = ie.load_network(
            network=self.net,
            device_name=self.config['device_name']
        )

        with open(
            os.path.join(self.config['model_path'], 'coco.names'), 'r'
        ) as f:
            self.labels_map = [x.strip() for x in f]

    @timing
    def recognize(self, frame):
        proc_image = cv2.resize(
            frame,
            (self.h, self.w),
            interpolation=cv2.INTER_LINEAR
        )
        # Change data layout from HWC to CHW
        proc_image = proc_image.transpose((2, 0, 1))

        try:
            result = self.exec_net.infer({self.input_blob: proc_image})
        except Exception:
            logger.exception("Exec Network is down")
            # Reset usb device. Find ids with lsusb
            if (
                self.config['device_name'] == 'MYRIAD'
                and self.config.get('id_vendor') is not None
                and self.config.get('id_product') is not None
            ):
                dev = finddev(
                    idVendor=self.config['id_vendor'],
                    idProduct=self.config['id_product']
                )
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

        objects = filter_objects(
            objects,
            iou_threshold=0.4,
            prob_threshold=PROB_THRESHOLD
        )

        def _add_object_label_and_total_area(e):
            e['label'] = self.labels_map[e['class_id']]
            e['total_area'] = (e['ymax'] - e['ymin']) * (e['xmax'] - e['xmin'])
            return e

        objects = list(map(_add_object_label_and_total_area, objects))

        return objects
