from typing import List
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.net = cv2.dnn.readNet(model_path)
        self.confidence_threshold = confidence_threshold
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, image: np.ndarray) -> List[dict]:
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        detected_objects = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            detected_objects.append({
                'box': box,
                'confidence': confidences[i],
                'class_id': class_ids[i]
            })

        return detected_objects