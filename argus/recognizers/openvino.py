import cv2
import logging
import ngraph as ng
import os
import sys

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
            num_requests=self.threads_count * 2
        )

        with open(os.path.join(self.config['model_path'], 'coco.names'), 'r') as f:
            self.labels_map = [x.strip() for x in f]


    def proc_image(self, frame):
        proc_frame = cv2.resize(
            frame,
            (self.h, self.w),
            interpolation=cv2.INTER_LINEAR
        )
        # Change data layout from HWC to CHW
        proc_frame = proc_frame.transpose((2, 0, 1))
        return proc_frame


    def get_filtered_objects_with_label(self, frame, result):
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


    @timing
    def split_and_recocnize(self, frame, thread_number):

        # Two request for one image (for left and right tail)
        if thread_number == 0:
            r1, r2 = thread_number, thread_number + 1
        else:
            r1, r2 = thread_number + 1, thread_number + 2

        h, w = frame.shape[0], frame.shape[1]  # e.g. 1080x1920
        # 960
        half_frame = int(w/2)
        # [120:1080, 0:960]
        left_frame = frame[h-half_frame:h, 0:half_frame]
        # [120:1080, 960:1920]
        right_frame = frame[h-half_frame:h, half_frame:w]

        proc_image_left = self.proc_image(left_frame)
        proc_image_right = self.proc_image(right_frame)

        try:
            self.exec_net.requests[r1].async_infer({self.input_blob: proc_image_left})
            self.exec_net.requests[r2].async_infer({self.input_blob: proc_image_right})
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

        result = {}
        output_queue = [r1, r2]
        while True:
            for i in output_queue:
                infer_status= self.exec_net.requests[i].wait(0)
                if infer_status == StatusCode.RESULT_NOT_READY:
                    continue
                if infer_status != StatusCode.OK:
                    logger.error("Unexpected infer status code")
                    sys.exit(0)
                result[i] = self.exec_net.requests[i].output_blobs
                output_queue.remove(i)
            if len(output_queue) == 0:
                break

        left_frame_objects = self.get_filtered_objects_with_label(left_frame, result[r1])
        right_frame_objects = self.get_filtered_objects_with_label(right_frame, result[r2])

        for obj in left_frame_objects:
            obj['ymin'] += h - half_frame
            obj['ymax'] += h - half_frame

        for obj in right_frame_objects:
            obj['xmin'] += half_frame
            obj['ymin'] += h - half_frame
            obj['xmax'] += half_frame
            obj['ymax'] += h - half_frame

        return left_frame_objects + right_frame_objects
