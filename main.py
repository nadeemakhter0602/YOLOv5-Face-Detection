from detector import Detector
import cv2
import time

def draw_bounding_boxes(image, boxes):
    for box in boxes:
        # overlay rectangle boxes and class name 
        cv2.rectangle(image, box, (0, 255, 255), 5)

# use yolov5n6 model
# need cudnn and CUDA toolkit installed for cuda as ONNXRuntime backend
detector_yolov5n6 = Detector('best_yolov5n6.onnx', 1/255.0, (416, 416), backend='cuda')
image = cv2.imread('assets/detect.jpg')
start = time.time()
boxes = detector_yolov5n6.detect(image)
draw_bounding_boxes(image, boxes)
cv2.imwrite('assets/inference_yolov5n6.jpg', image)
print('YOLOv5n6 took', time.time() - start, 'seconds')

# use yolov5s6 model
detector_yolov5s6 = Detector('best_yolov5s6.onnx', 1/255.0, (640, 640), backend='cuda')
image = cv2.imread('assets/detect.jpg')
start = time.time()
boxes = detector_yolov5n6.detect(image)
draw_bounding_boxes(image, boxes)
cv2.imwrite('assets/inference_yolov5s6.jpg', image)
print('YOLOv5s6 took', time.time() - start, 'seconds')
