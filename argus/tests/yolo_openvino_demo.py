import numpy as np
import cv2, time

from openvino.runtime import Core


MODEL_NAME = "yolov8s"

with open('../models/coco.names', 'r') as f:
    CLASSES = [x.strip() for x in f]

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 实例化Core对象
core = Core() 
# 载入并编译模型
net = core.compile_model(f'../models/{MODEL_NAME}.xml', device_name="AUTO")
# 获得模型输出节点
output_node = net.outputs[0]  # yolov8n只有一个输出节点
ir = net.create_infer_request()


start = time.time()

# ==============================
frame = cv2.imread("../res/test.jpeg")

[height, width, _] = frame.shape
length = max((height, width))
image = np.zeros((length, length, 3), np.uint8)
image[0:height, 0:width] = frame
scale = length / 640

blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

# import pdb; pdb.set_trace()

outputs = ir.infer(blob)[output_node]



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
for i in range(len(result_boxes)):
    index = result_boxes[i]
    box = boxes[index]
    detection = {
        'class_id': class_ids[index],
        'class_name': CLASSES[class_ids[index]],
        'confidence': scores[index],
        'box': box,
        'scale': scale}
    detections.append(detection)
    draw_bounding_box(frame, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                      round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))



end = time.time()
# show FPS
fps = (1 / (end - start)) 
fps_label = "Throughput: %.2f FPS" % fps
cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('YOLOv8 OpenVINO Infer Demo on AIxBoard', frame)
cv2.waitKey()
cv2.destroyAllWindows()