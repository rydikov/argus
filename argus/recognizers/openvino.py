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

        ie = IECore()
        self.net = ie.read_network(
            os.path.join(self.config['model_path'], 'frozen_darknet_yolov4_model.xml'),
            os.path.join(self.config['model_path'], 'frozen_darknet_yolov4_model.bin')
        )

        self.function_from_cnn = ng.function_from_cnn(self.net)

        # Extract network params
        self.input_blob = next(iter(self.net.input_info))
        _, _, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape

        if isinstance(self.config['device_name'], list) and len(self.config['device_name']) == 2:
            self.exec_net1 = ie.load_network(
                network=self.net,
                device_name=self.config['device_name'][0],
                num_requests=self.threads_count
            )
            self.exec_net2 = ie.load_network(
                network=self.net,
                device_name=self.config['device_name'][1],
                num_requests=self.threads_count
            )
        elif isinstance(self.config['device_name'], str):
            self.exec_net1 = self.exec_net2 = ie.load_network(
                network=self.net,
                device_name=self.config['device_name'],
                num_requests=self.threads_count * 2
            )
        else:
            logger.error('One or two devices for recognize allowed')
            sys.exit(1)



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

    def get_net_result(self, net, request_num):
        net.requests[request_num].wait()
        return net.requests[request_num].output_blobs


    @timing
    def split_and_recocnize(self, frame, thread_number):

        if isinstance(self.config['device_name'], str):
            if thread_number == 0:
                request_num_for_left_frame = thread_number
                request_num_for_right_frame = thread_number + 1
            else:
                request_num_for_left_frame = thread_number + 1
                request_num_for_right_frame = thread_number + 2
        else:
            request_num_for_left_frame = thread_number
            request_num_for_right_frame = thread_number


        h, w = frame.shape[0], frame.shape[1]  # e.g. 1080x1920
        # 960
        half_frame = int(w/2)
        # [120:1080, 0:960]
        left_frame = frame[h-half_frame:h, 0:half_frame]
        # [120:1080, 960:1920]
        right_frame = frame[h-half_frame:h, half_frame:w]

        proc_image_left = self.proc_image(left_frame)
        proc_image_right = self.proc_image(right_frame)

        asyncio.run(
            self.create_infer_tasks(
                proc_image_left, 
                request_num_for_left_frame,
                proc_image_right,
                request_num_for_right_frame
            )
        )
        
        result_for_left_frame = self.get_net_result(self.exec_net1, request_num_for_left_frame)
        result_for_right_frame = self.get_net_result(self.exec_net2, request_num_for_right_frame)

        left_frame_objects = self.get_filtered_objects_with_label(left_frame, result_for_left_frame)
        right_frame_objects = self.get_filtered_objects_with_label(right_frame, result_for_right_frame)

        for obj in left_frame_objects:
            obj['ymin'] += h - half_frame
            obj['ymax'] += h - half_frame

        for obj in right_frame_objects:
            obj['xmin'] += half_frame
            obj['ymin'] += h - half_frame
            obj['xmax'] += half_frame
            obj['ymax'] += h - half_frame

        return left_frame_objects + right_frame_objects


    async def create_infer_tasks(self, left_frame, request_num_for_left_frame, right_frame, request_num_for_right_frame):
        await asyncio.gather(
            self.infer(self.exec_net1, request_num_for_left_frame, left_frame),
            self.infer(self.exec_net2, request_num_for_right_frame, right_frame),
        )

    async def infer(self, exec_net, request_num, frame):
        print(">>>>>>>>")
        exec_net.requests[request_num].async_infer({self.input_blob: frame})
        # await asyncio.sleep(10)
        print("<<<<<<<<")
