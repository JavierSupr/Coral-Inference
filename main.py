import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageDraw
import time

class ObjectDetector:

    def __init__(self, model_path, label_path, threshold=0.3):
        self.threshold = threshold
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:]
        self.class_labels = self.load_class_labels(label_path)

    def load_class_labels(self, file_path):
        with open(file_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def preprocess_frame(self, frame):
        image_resized = frame.resize((self.input_shape[0], self.input_shape[1]))
        image_float32 = np.array(image_resized).astype(np.float32)
        image_normalized = image_float32 / 255.0
        image_with_batch = np.expand_dims(image_normalized, 0)
        return image_with_batch

    def nms(self, boxes, scores, threshold):
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            overlap = (w * h) / areas[order[1:]]
            inds = np.where(overlap <= threshold)[0]
            order = order[inds + 1]
        return keep

    def perform_inference(self, frame, preprocessed_frame, orig_h, orig_w):
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_frame)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        output = np.copy(output_data[0])

        boxes = []
        scores = []

        for i in range(output.shape[1]):
            detection = output[:, i]
            x_center, y_center, width, height = detection[:4]
            x1 = max(0, int((x_center - width / 2) * orig_w))
            y1 = max(0, int((y_center - height / 2) * orig_h))
            x2 = min(orig_w - 1, int((x_center + width / 2) * orig_w))
            y2 = min(orig_h - 1, int((y_center + height / 2) * orig_h))
            score = np.max(detection[4:])
            boxes.append([x1, y1, x2, y2])
            scores.append(score)

        boxes = np.array(boxes)
        scores = np.array(scores)
        keep = self.nms(boxes, scores, 0.3)

        results = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            score = scores[idx]
            detection = output[:, idx]
            cls = np.argmax(output[:, idx][4:])
            if score >= self.threshold:
                class_name = self.class_labels[cls]
                results.append((class_name, score, (x1, y1, x2, y2)))
        return results

    def run_detection(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            orig_w, orig_h = frame_pil.size
            preprocessed_frame = self.preprocess_frame(frame_pil)

            detections = self.perform_inference(frame_pil, preprocessed_frame, orig_h, orig_w)

            print("Detections:")
            for det in detections:
                class_name, score, bbox = det
                print(f"Class: {class_name}, Score: {score:.2f}, BBox: {bbox}")

        cap.release()
        print("Detection completed.")

if __name__ == "__main__":
    detector = ObjectDetector(
        model_path='best_full_integer_quant_edgetpu.tflite',
        label_path='label.txt'
    )
    video_path = '333 VID_20231011_170120.mp4'
    detector.run_detection(video_path)
