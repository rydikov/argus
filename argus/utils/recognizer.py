import cv2
import logging
import os
import sys
import numpy as np

from openvino.inference_engine import IECore, StatusCode
from usb.core import find as finddev

from argus.utils.timing import timing
from argus.utils.yolo import get_objects, filter_objects

PROB_THRESHOLD = 0.85

logger = logging.getLogger('json')


def letterbox(img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    w, h = size

    # Scale ratio (new / old)
    r = min(h / shape[0], w / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (w, h)
        ratio = w / shape[1], h / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    top2, bottom2, left2, right2 = 0, 0, 0, 0
    if img.shape[0] != h:
        top2 = (h - img.shape[0])//2
        bottom2 = top2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    elif img.shape[1] != w:
        left2 = (w - img.shape[1])//2
        right2 = left2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


class OpenVinoRecognizer:

    def __init__(self, net_config):
        self.net_config = net_config

        self.frame_buffer = {}

        models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'models'))

        self.ie = IECore()
        self.net = self.ie.read_network(
            os.path.join(models_path, 'yolov8n.xml'),
            os.path.join(models_path, 'yolov8n.bin')
        )

        # Extract network params
        self.input_blob = next(iter(self.net.input_info))
        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape

        self.exec_net = self.ie.load_network(
            network=self.net,
            device_name=self.net_config['device_name'],
            num_requests=self.net_config['num_requests'],
        )

        #self._thermal_metric_support = 'DEVICE_THERMAL' in self.ie.get_metric(self.net_config['device_name'], 'SUPPORTED_METRICS')

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

        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(640, 640), swapRB=True)

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


    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = f'{self.labels_map[class_id]} ({confidence:.2f})'
        color = (255,255,255)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 1)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)

    def get_result(self, request_id):

        infer_status = self.exec_net.requests[request_id].wait(0)

        if (
            infer_status == StatusCode.RESULT_NOT_READY
            or self.frame_buffer.get(request_id) is None
        ):
            logger.warn("Unable to get result for {}, status {}".format(request_id, infer_status))
            return None

        result = self.exec_net.requests[request_id].output_blobs

        buffer_item = self.frame_buffer.pop(request_id)

        ##

        outputs = result['output0'].buffer

        outputs = np.array([cv2.transpose(outputs[0])])
        
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []

        height, width, _ = buffer_item.frame.shape
        length = max((height, width))
        scale = length/640

        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                'class_name': self.labels_map[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': scale}
            detections.append(detection)

            self.draw_bounding_box(
                buffer_item.frame, 
                class_ids[index], 
                scores[index], 
                round(box[0] * scale), 
                round(box[1] * scale),
                round((box[0] + box[2]) * scale), 
                round((box[1] + box[3]) * scale)
            )

        # objects = get_objects(
        #     result,
        #     self.net,
        #     (self.h, self.w),
        #     (buffer_item.frame.shape[0], buffer_item.frame.shape[1]),
        #     PROB_THRESHOLD,
        # )

        # objects = filter_objects(
        #     objects,
        #     iou_threshold=0.4,
        #     prob_threshold=PROB_THRESHOLD
        # )

        # buffer_item.map_objects_to_frame(objects, self.labels_map)
        if detections:
            buffer_item.objects_detected = True
            buffer_item.important_objects_detected = True

        return buffer_item
