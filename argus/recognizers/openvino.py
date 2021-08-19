import cv2
import logging
import ngraph as ng
import os
import sys
import asyncio

from openvino.inference_engine import IECore, StatusCode
from usb.core import find as finddev

from argus.helpers.timing import timing
from argus.helpers.yolo import get_objects, filter_objects

PROB_THRESHOLD = 0.4

logger = logging.getLogger('json')


class OpenVinoRecognizer:

    def __init__(self, config, threads_count):
        self.config = config
        self.threads_count = threads_count
        self.frame_buffer = {}

        ie = IECore()
        self.net = ie.read_network(
            os.path.join(self.config['model_path'], 'frozen_darknet_yolov4_model.xml'),
            os.path.join(self.config['model_path'], 'frozen_darknet_yolov4_model.bin')
        )

        self.function_from_cnn = ng.function_from_cnn(self.net)

        # Extract network params
        self.input_blob = next(iter(self.net.input_info))
        _, _, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape

        self.exec_net = ie.load_network(
            network=self.net,
            device_name=self.config['device_name'],
            num_requests=self.config['num_requests'],
        )

        with open(os.path.join(self.config['model_path'], 'coco.names'), 'r') as f:
            self.labels_map = [x.strip() for x in f]

    @timing
    def wait(self):
        return self.exec_net.wait(num_requests=1)


    def get_request_id(self):
        is_waited = False
        infer_request_id = self.exec_net.get_idle_request_id()
        if infer_request_id < 0:
            status = self.wait()
            is_waited = True
            if status != StatusCode.OK:
                raise Exception("Wait for idle request failed!")
            infer_request_id = self.exec_net.get_idle_request_id()
            if infer_request_id < 0:
                raise Exception("Invalid request id!")
        return infer_request_id, is_waited


    def send_to_recocnize(self, frame, request_id):

        self.frame_buffer[request_id] = frame

        proc_frame = cv2.resize(
            frame,
            (self.h, self.w),
            interpolation=cv2.INTER_LINEAR
        )
        # Change data layout from HWC to CHW
        proc_frame = proc_frame.transpose((2, 0, 1))

        try:
            self.exec_net.requests[request_id].async_infer({self.input_blob: proc_frame})
        except Exception:
            logger.exception("Exec Network is down")
            # Reset usb device. Find ids with lsusb
            if (
                'MYRIAD' in self.config['device_name']
                and self.config.get('id_vendor') is not None
                and self.config.get('id_product') is not None
            ):
                dev = finddev(
                    idVendor=self.config['id_vendor'],
                    idProduct=self.config['id_product']
                )
                dev.reset()
            sys.exit(0)


    def get_result(self, request_id):
        infer_status= self.exec_net.requests[request_id].wait(0)

        if infer_status == StatusCode.RESULT_NOT_READY:
            return [], None

        frame = self.frame_buffer[request_id]
        result = self.exec_net.requests[request_id].output_blobs

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

        result = list(map(_add_object_label_and_total_area, objects))

        return result, frame