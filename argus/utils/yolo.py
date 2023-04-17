# From examles
import numpy as np

from math import exp as exp
from argus.utils.timing import timing


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self,  side):
        self.num = 3
        self.coords = 4
        self.classes = 80
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0]


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, resized_im_h=640, resized_im_w=640):
    gain = min(resized_im_w / im_w, resized_im_h / im_h)  # gain  = old / new
    pad = (resized_im_w - im_w * gain) / 2, (resized_im_h - im_h * gain) / 2  # wh padding
    x = int((x - pad[0])/gain)
    y = int((y - pad[1])/gain)

    w = int(width/gain)
    h = int(height/gain)
 
    xmin = max(0, int(x - w / 2))
    ymin = max(0, int(y - h / 2))
    xmax = min(im_w, int(xmin + w))
    ymax = min(im_h, int(ymin + h))
    # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
    # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())


@timing
def parse_yolo_region_legacy(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------    
    out_blob_n, out_blob_c, out_blob_h, out_blob_w = blob.shape
    predictions = 1.0/(1.0+np.exp(-blob)) 
                   
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
 
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    bbox_size = int(out_blob_c/params.num) #4+1+num_classes

    for row, col, n in np.ndindex(params.side, params.side, params.num):
        bbox = predictions[0, n*bbox_size:(n+1)*bbox_size, row, col]
        
        x, y, width, height, object_probability = bbox[:5]
        class_probabilities = bbox[5:]
        if object_probability < threshold:
            continue
        x = (2*x - 0.5 + col)*(resized_image_w/out_blob_w)
        y = (2*y - 0.5 + row)*(resized_image_h/out_blob_h)
        if int(resized_image_w/out_blob_w) == 8 & int(resized_image_h/out_blob_h) == 8: #80x80, 
            idx = 0
        elif int(resized_image_w/out_blob_w) == 16 & int(resized_image_h/out_blob_h) == 16: #40x40
            idx = 1
        elif int(resized_image_w/out_blob_w) == 32 & int(resized_image_h/out_blob_h) == 32: # 20x20
            idx = 2

        width = (2*width)**2* params.anchors[idx * 6 + 2 * n]
        height = (2*height)**2 * params.anchors[idx * 6 + 2 * n + 1]
        class_id = np.argmax(class_probabilities)
        confidence = object_probability
        objects.append(scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
                                  im_h=orig_im_h, im_w=orig_im_w, resized_im_h=resized_image_h, resized_im_w=resized_image_w))
    return objects


@timing
def parse_yolo_region(
    blob: np.ndarray,
    resized_image_shape: tuple[int, int],
    original_im_shape: tuple[int, int],
    params,
    threshold: float
):
    # Vector optimization
    out_blob_n, out_blob_c, out_blob_h, out_blob_w = blob.shape
    predictions = 1.0 / (1.0 + np.exp(-blob))
    predictions = predictions.squeeze()

    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    assert out_blob_c % params.num == 0
    bbox_size = int(out_blob_c / params.num)  # 4+1+num_classes

    predictions = np.transpose(predictions, (1, 2, 0))
    indexes = [n * bbox_size + 4 for n in range(params.num)]
    matched_idexes = np.nonzero(predictions[:, :, indexes] >= threshold)
    
    for row, col, n in zip(*matched_idexes):
        bbox = predictions[row, col, n * bbox_size:(n + 1) * bbox_size]
        x, y, width, height, object_probability = bbox[:5]
        class_probabilities = bbox[5:]

        x = (2 * x - 0.5 + col) * (resized_image_w / out_blob_w)
        y = (2 * y - 0.5 + row) * (resized_image_h / out_blob_h)
        if int(resized_image_w / out_blob_w) == 8 & int(
                resized_image_h / out_blob_h) == 8:  # 80x80,
            idx = 0
        elif int(resized_image_w / out_blob_w) == 16 & int(
                resized_image_h / out_blob_h) == 16:  # 40x40
            idx = 1
        elif int(resized_image_w / out_blob_w) == 32 & int(
                resized_image_h / out_blob_h) == 32:  # 20x20
            idx = 2

        width = (2 * width) ** 2 * params.anchors[idx * 6 + 2 * n]
        height = (2 * height) ** 2 * params.anchors[idx * 6 + 2 * n + 1]
        class_id = np.argmax(class_probabilities)
        confidence = object_probability

        objects.append(scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
                                  im_h=orig_im_h, im_w=orig_im_w, resized_im_h=resized_image_h, resized_im_w=resized_image_w))
    return objects

def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def get_objects(output, net, new_frame_height_width, source_height_width, prob_threshold):

    objects = list()

    for _, out_blob in output.items():
        
        layer_params = YoloParams(side=out_blob.buffer.shape[2])
        objects += parse_yolo_region(
            out_blob.buffer,
            new_frame_height_width,
            source_height_width,
            layer_params,
            prob_threshold
        )

    return objects


def filter_objects(objects, iou_threshold, prob_threshold):
    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    objects = sorted(objects, key=lambda obj: obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    return tuple(obj for obj in objects if obj['confidence'] >= prob_threshold)
