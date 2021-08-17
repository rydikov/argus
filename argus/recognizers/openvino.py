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
            self.exec_net = []
            for device in config['device_name']:
                exec_net = ie.load_network(
                    network=self.net,
                    device_name=device,
                    num_requests=self.threads_count
                )
                self.exec_net.append(exec_net)

        elif isinstance(self.config['device_name'], str):
            self.exec_net = ie.load_network(
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


    def get_filtered_objects_with_label(self, frame_height_width, result):

        objects = get_objects(
            result,
            self.net,
            (self.h, self.w),
            (frame_height_width, frame_height_width),
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

        h, w = frame.shape[0], frame.shape[1]  # e.g. 1080x1920
        # 960
        frame_height_width = int(w/2) # half_frame
        # [120:1080, 0:960]
        left_frame = frame[h-frame_height_width:h, 0:frame_height_width]
        # [120:1080, 960:1920]
        right_frame = frame[h-frame_height_width:h, frame_height_width:w]

        proc_frame_left = self.proc_image(left_frame)
        proc_frame_right = self.proc_image(right_frame)

        if isinstance(self.exec_net, list):
            infer_method = self.multi_devices_infer
        else:
            infer_method = self.single_device_infer

        result = infer_method(proc_frame_left, proc_frame_right, thread_number)

        left_frame_objects = self.get_filtered_objects_with_label(frame_height_width, result['left_frame'])
        right_frame_objects = self.get_filtered_objects_with_label(frame_height_width, result['right_frame'])

        for obj in left_frame_objects:
            obj['ymin'] += h - frame_height_width
            obj['ymax'] += h - frame_height_width

        for obj in right_frame_objects:
            obj['xmin'] += frame_height_width
            obj['ymin'] += h - frame_height_width
            obj['xmax'] += frame_height_width
            obj['ymax'] += h - frame_height_width

        return left_frame_objects + right_frame_objects


    def try_infer(self, exec_net, request_num, frame):
        try:
            exec_net.requests[request_num].async_infer({self.input_blob: frame})
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
    
    def single_device_infer(self, proc_image_left, proc_image_right, thread_number):
        # Two request for one image (for left and right part)
        if thread_number == 0:
            r1, r2 = thread_number, thread_number + 1
        else:
            r1, r2 = thread_number + 1, thread_number + 2

        self.try_infer(self.exec_net, r1, proc_image_left)
        self.try_infer(self.exec_net, r2, proc_image_right)

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
        
        result['left_frame'] = result.pop(r1)
        result['right_frame'] = result.pop(r2)

        return result

    #####
    def multi_devices_infer(self, proc_image_left, proc_image_right, thread_number):
        
        def _get_net_result(net, request_num):
            net.requests[request_num].wait()
            return net.requests[request_num].output_blobs

        result = {}
        asyncio.run(self.create_infer_tasks(proc_image_left, proc_image_right, thread_number))
        result['left_frame'] = _get_net_result(self.exec_net[0], thread_number)
        result['right_frame'] = _get_net_result(self.exec_net[1], thread_number)
        return result

    async def create_infer_tasks(self, left_frame, right_frame, thread_number):
        await asyncio.gather(
            self.infer(self.exec_net[0], thread_number, left_frame),
            self.infer(self.exec_net[1], thread_number, right_frame),
        )

    async def infer(self, exec_net, request_num, frame):
        self.try_infer(exec_net, request_num, frame)
    #####