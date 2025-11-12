import numpy as np
import cv2

class DepthEstimator:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        # Load the pre-trained depth estimation model
        model = cv2.dnn.readNet(model_path)
        return model

    def estimate_depth(self, image):
        # Prepare the image for depth estimation
        blob = cv2.dnn.blobFromImage(image, 1.0, (640, 480), (104.0, 177.0, 123.0), swapRB=True, crop=False)
        self.model.setInput(blob)
        depth_map = self.model.forward()
        return depth_map

    def post_process_depth(self, depth_map):
        # Normalize and resize the depth map for visualization
        depth_map = depth_map.squeeze()
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = cv2.resize(depth_map, (640, 480))
        return depth_map.astype(np.uint8)