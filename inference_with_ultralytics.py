import cv2
from ultralytics import YOLO
import argparse

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='YOLO Inference on Coral Dev Board')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to YOLO TFLite model')
    parser.add_argument('--source', type=str, required=True, 
                        help='Path to input video file')
    args = parser.parse_args()

    # Load YOLO model
    try:
        model = YOLO(args.model)
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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model.predict(frame, verbose=False)

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

    # Release resources
    cap.release()
    print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    main()