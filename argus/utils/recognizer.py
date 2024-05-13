import cv2
import logging
import os
import sys
import numpy as np

from openvino.inference_engine import IECore, StatusCode
from usb.core import find as finddev

from argus.utils.timing import timing

PROB_THRESHOLD = 0.25

logger = logging.getLogger('json')


class OpenVinoRecognizer:

    def __init__(self, net_config):
        self.net_config = net_config

        self.frame_buffer = {}

        models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'models'))
        model = self.net_config.get('model', 'yolov8s')

        self.ie = IECore()
        self.net = self.ie.read_network(
            os.path.join(models_path, f'{model}.xml')
        )

        # Extract network params
        self.input_blob = next(iter(self.net.input_info))
        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape

        self.exec_net = self.ie.load_network(
            network=self.net,
            device_name=self.net_config['device_name'],
            num_requests=self.net_config['num_requests'],
        )

        with open(os.path.join(models_path, 'coco.names'), 'r') as f:
            self.labels_map = [x.strip() for x in f]

    def log_temperature(self):

        devices = []
        if 'MULTI' in self.net_config['device_name']:
            _, devices = self.net_config['device_name'].split(':')
            devices = devices.split(',')
        elif 'MYRIAD' in self.net_config['device_name']:
            devices = self.net_config['device_name']
       
        for device in devices:
            themperature = self.ie.get_metric(device, 'DEVICE_THERMAL')
            logger.info(
                "Device {} themperature: {}".format(device, themperature), 
                extra={'device': device, 'themperature': themperature}
            )

    @timing
    def wait(self):
        return self.exec_net.wait(num_requests=1)

    def get_request_id(self, need_wait=False):
        """
        If need_wait is True – we wait for the result. Else noblock skip.
        """
        infer_request_id = self.exec_net.get_idle_request_id()
        if infer_request_id < 0:
            if need_wait:
                status = self.wait()
                if status != StatusCode.OK:
                    raise Exception("Wait for idle request failed!")
                infer_request_id = self.exec_net.get_idle_request_id()
                if infer_request_id < 0:
                    raise Exception("Invalid request id!")
            else:
                return None
        return infer_request_id

    def send_to_recocnize(self, queue_item, request_id):

        self.frame_buffer[request_id] = queue_item

        frame = queue_item.frame

        height, width, _ = frame.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = frame

        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(self.h, self.w), swapRB=True)

        try:
            self.exec_net.requests[request_id].async_infer({self.input_blob: blob})
        except Exception:
            logger.exception("Exec Network is down")
            # Reset usb device. Find ids with lsusb
            if (
                'MYRIAD' in self.net_config['device_name']
                and self.net_config.get('id_vendor') is not None
                and self.net_config.get('id_product') is not None
            ):
                dev = finddev(
                    idVendor=self.net_config['id_vendor'],
                    idProduct=self.net_config['id_product']
                )
                dev.reset()
            sys.exit(0)


    def get_result(self, request_id):

        infer_status = self.exec_net.requests[request_id].wait(0)

        if (
            infer_status == StatusCode.RESULT_NOT_READY
            or self.frame_buffer.get(request_id) is None
        ):
            logger.warning("Unable to get result for {}, status {}".format(request_id, infer_status))
            return None

        result = self.exec_net.requests[request_id].output_blobs

        buffer_item = self.frame_buffer.pop(request_id)

        detections = []

        for _, out_blob in result.items():

            outputs = np.array([cv2.transpose(out_blob.buffer[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []

            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                if maxScore >= PROB_THRESHOLD:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]), 
                        outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                        outputs[0][i][2], 
                        outputs[0][i][3]
                    ]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, PROB_THRESHOLD, nms_threshold=0.45, eta=0.5)

            height, width, _ = buffer_item.frame.shape
            length = max((height, width))
            scale = length / self.w

            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    'class_id': class_ids[index],
                    'label': self.labels_map[class_ids[index]],
                    'confidence': scores[index],
                    'xmin': round(box[0] * scale), 
                    'ymin': round(box[1] * scale),
                    'xmax': round((box[0] + box[2]) * scale), 
                    'ymax': round((box[1] + box[3]) * scale)
                }

                detections.append(detection)

        buffer_item.map_detections_to_frame(detections)
        buffer_item.mark_as_recognized()

        return buffer_item
