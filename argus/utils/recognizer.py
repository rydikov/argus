import cv2
import logging
import os
import sys
import numpy as np

from datetime import datetime, timedelta
from openvino.runtime import Core, AsyncInferQueue

from argus.utils.timing import timing
from argus.globals import (
    SILENT_TIME, 
    last_detection, 
    last_frame_save_time, 
    silent_notify_until_time,
    send_frames_after_signal
)

LOG_RECOGNIZE_RPS_TIME = timedelta(minutes=1)

PROB_THRESHOLD = 0.35

logger = logging.getLogger('json')

# Recognize RPS
recognize_rps = {'count': 0, 'time': datetime.now()}
def up_rps():
    recognize_rps['count'] += 1
    now = datetime.now()
    if recognize_rps['time'] + LOG_RECOGNIZE_RPS_TIME < now:
        delta = now - recognize_rps['time']
        rps = recognize_rps['count'] / delta.seconds
        recognize_rps['time'] = now
        recognize_rps['count'] = 0
        logger.info(f'Recognized RPS: {rps}', extra={'rps': rps})



class OpenVinoRecognizer:

    def __init__(self, net_config, telegram):
        self.net_config = net_config
        self.telegram = telegram

        models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'models'))
        model_name = self.net_config.get('model', 'yolov8s')

        self.core = Core()
        model = self.core.read_model(os.path.join(models_path, f'{model_name}.xml'))

        # Extract network params
        self.input_layer_ir = model.input(0)
        self.n, self.c, self.h, self.w = self.input_layer_ir.shape

        compiled_model = self.core.compile_model(model, self.net_config['device_name'])
        self.ireqs = AsyncInferQueue(compiled_model, self.net_config['num_requests'])
        self.ireqs.set_callback(self.process_frame)

        logger.info(
            f'OPTIMAL_NUMBER_OF_INFER_REQUESTS: {compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")}'
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
            themperature = self.core.get_property(device, 'DEVICE_THERMAL')
            logger.info(
                "Device {} themperature: {}".format(device, themperature), 
                extra={'device': device, 'themperature': themperature}
            )

    def send_to_recognize(self, queue_item):

        frame = queue_item.frame

        height, width, _ = frame.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = frame

        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(self.h, self.w), swapRB=True)

        try:
            self.ireqs.start_async({self.input_layer_ir.any_name: blob}, queue_item)
        except Exception:
            logger.exception("Exec Network is down")
            if self.telegram is not None:
                self.telegram.send_message(f'Exec Network is down')
            sys.exit(0)


    def process_frame(self, infer_request, queue_item):

        up_rps()
        
        result = infer_request.get_output_tensor(0).data

        detections = []

        outputs = np.array([cv2.transpose(result[0])])
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

        height, width, _ = queue_item.frame.shape
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

        if detections:
            queue_item.map_detections_to_frame(detections)   
        queue_item.mark_as_recognized()
        thread_name = queue_item.thread_name

        if queue_item.objects_detected:
            # Save detected frame every 1 sec
            delta = timedelta(seconds=1)
            prefix = 'detected'
            last_detection[thread_name] = datetime.now()
        else:
            # Save frame every N (save_every_sec) sec
            delta = timedelta(seconds=queue_item.save_every_sec)
            prefix = None
            
        if last_frame_save_time.get(thread_name, datetime.now() - delta) + delta <= datetime.now():
            queue_item.save(prefix=prefix)
            last_frame_save_time[thread_name] = datetime.now()
            # Telegram alerting
            if (
                queue_item.important_objects_detected and
                self.telegram is not None and
                last_detection[thread_name] > silent_notify_until_time.get(thread_name, datetime.now() - SILENT_TIME)
            ):
                self.telegram.send_message(f'Objects detected: {queue_item.url}')
                silent_notify_until_time[thread_name] = datetime.now() + SILENT_TIME

        # Send frame to telegram after external signal
        if thread_name in send_frames_after_signal and self.telegram is not None:
            send_frames_after_signal.remove(thread_name)
            self.telegram.send_frame(queue_item.frame, f'Photo from {thread_name}')
