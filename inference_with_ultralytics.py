import cv2
from ultralytics import YOLO
import argparse
import numpy as np

def preprocess_frame(frame, input_size=(640, 640)):
    """Preprocess frame for TFLite model input."""
    # Resize frame while maintaining aspect ratio
    resized = cv2.resize(frame, input_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0,1] if needed
    normalized = resized / 255.0
    
    # Add batch dimension
    input_data = np.expand_dims(normalized, axis=0)
    
    return input_data

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='YOLO Inference on Coral Dev Board')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to YOLO TFLite Edge TPU model')
    parser.add_argument('--source', type=str, required=True, 
                        help='Path to input video file')
    args = parser.parse_args()

    # Load YOLO model with explicit task
    try:
        model = YOLO(args.model, task='detect')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open video capture
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.source}")
        return

    # Process video
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            processed_frame = preprocess_frame(frame)

            # Run inference with explicit preprocessing
            results = model.predict(processed_frame, verbose=False)

            # Print detection results
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

    finally:
        # Release resources
        cap.release()
        print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    main()