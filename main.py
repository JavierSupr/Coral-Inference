import argparse
import cv2
import os
import time
import numpy as np
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def yolo_post_process(output_tensor, labels, threshold):
    predictions = np.squeeze(output_tensor)
    results = []
    for pred in predictions.T:
        class_scores = pred[4:]
        max_class_score = np.max(class_scores)
        class_id = np.argmax(class_scores)
        if max_class_score > threshold:
            bbox = pred[:4]
            results.append({
                "bbox": bbox,
                "class_id": class_id,
                "score": max_class_score,
                "label": labels.get(class_id, "Unknown")
            })
    return results

def draw_results(frame, results):
    height, width, _ = frame.shape
    for result in results:
        bbox = result["bbox"]
        label = result["label"]
        score = result["score"]
        xmin = int(bbox[0] * width)
        ymin = int(bbox[1] * height)
        xmax = int(bbox[2] * width)
        ymax = int(bbox[3] * height)
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(width, xmax), min(height, ymax)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label_text = f"{label} ({score:.2f})"
        cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def video_stream(args):
    print(f'Loading {args.model} with {args.labels}')
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    cap = cv2.VideoCapture(int(args.input)) if args.input.isdigit() else cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, inference_size)
        start_time = time.time()
        run_inference(interpreter, frame_resized.tobytes())
        inference_time = time.time() - start_time
        output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
        results = yolo_post_process(output_tensor, labels, args.threshold)
        print(f"\nFrame Processed in {inference_time:.2f} seconds")
        for result in results:
            print(f"- {result['label']}: {result['score']:.2f}, BBox: {result['bbox']}")
    cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n_full_integer_quant_edgetpu.tflite')
    parser.add_argument('--labels', default='label.txt')
    parser.add_argument('--input', type=str, default='333 VID_20231011_170120.mp4')
    parser.add_argument('--threshold', type=float, default=0.1)
    args = parser.parse_args()
    video_stream(args)

if __name__ == '__main__':
    main()
