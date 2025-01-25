import cv2
import numpy as np
from ultralytics import YOLO
import argparse

def preprocess_frame(frame, input_size=(640, 640)):
    """Robust frame preprocessing for TFLite model."""
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    scale = min(input_size[0] / width, input_size[1] / height)
    new_width = min(int(width * scale), input_size[0])
    new_height = min(int(height * scale), input_size[1])

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)

    y_offset = max(0, (input_size[1] - new_height) // 2)
    x_offset = max(0, (input_size[0] - new_width) // 2)

    try:
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    except Exception as e:
        print(f"Error during image placement: {e}")
        return None

    normalized = canvas.astype(np.float32) / 255.0
    input_data = np.expand_dims(normalized, axis=0)
    return input_data


def main():
    parser = argparse.ArgumentParser(description='YOLO Inference on Coral Dev Board')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to YOLO TFLite Edge TPU model')
    parser.add_argument('--source', type=str, required=True, 
                        help='Path to input video file')
    args = parser.parse_args()

    try:
        model = YOLO(args.model, task='detect')
    except Exception as e:
        print(f"Model load error: {e}")
        return

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Could not open video {args.source}")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        if processed_frame is None:
            continue

        try:
            results = model.predict(processed_frame, verbose=False)

            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    print(f"Frame {frame_count}:")
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        print(f"  Detected: {model.names[cls]} (Confidence: {conf:.2f})")

            frame_count += 1

        except Exception as e:
            print(f"Inference error: {e}")
            break

    cap.release()
    print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    main()