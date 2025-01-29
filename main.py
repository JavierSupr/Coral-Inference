import argparse
import cv2
import os
import time
import numpy as np
from flask import Flask, Response
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

app = Flask(__name__)

# Global variables
frame_with_results = None

def yolo_post_process(output_tensor, labels, threshold):
    """
    Process YOLO output tensor to extract bounding boxes, class scores, and labels.
    """
    predictions = np.squeeze(output_tensor)
    results = []
    for pred in predictions.T:  # Transpose to iterate over each prediction
        class_scores = pred[4:]  # Assuming the first 4 values are bounding box
        max_class_score = np.max(class_scores)
        class_id = np.argmax(class_scores)

        if max_class_score > threshold:
            bbox = pred[:4]  # First 4 are bounding box coordinates
            results.append({
                "bbox": bbox,
                "class_id": class_id,
                "score": max_class_score,
                "label": labels.get(class_id, "Unknown")
            })

    return results


def draw_results(frame, results):
    """
    Draw detection results on the frame.
    """
    height, width, _ = frame.shape  # Get frame dimensions

    for result in results:
        bbox = result["bbox"]
        label = result["label"]
        score = result["score"]

        # Convert YOLO's normalized bbox to absolute pixel values
        xmin = int(bbox[0] * width)
        ymin = int(bbox[1] * height)
        xmax = int(bbox[2] * width)
        ymax = int(bbox[3] * height)

        # Ensure bounding box within frame limits
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(width, xmax), min(height, ymax)

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Label text (with background for readability)
        label_text = f"{label} ({score:.2f})"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x, text_y = xmin, ymin - 10 if ymin - 10 > 10 else ymin + 10
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame



def video_stream(args):
    global frame_with_results
    print('Loading {} with {} labels.'.format(args.model, args.labels))
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

        # Convert frame to model's input size and perform inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, inference_size)

        start_time = time.time()
        run_inference(interpreter, frame_resized.tobytes())
        inference_time = time.time() - start_time

        # Extract results and update frame with bounding boxes
        output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
        results = yolo_post_process(output_tensor, labels, args.threshold)

        # Draw results on the frame
        frame_with_results = draw_results(frame, results)  # Convert back to BGR format

        # Print detection results for debugging
        print(f"\nFrame Processed in {inference_time:.2f} seconds")
        for result in results:
            print(f"- {result['label']}: {result['score']:.2f}, BBox: {result['bbox']}")

    cap.release()



@app.route('/video_feed')
def video_feed():
    """
    Video streaming route.
    """
    def generate():
        global frame_with_results
        while True:
            if frame_with_results is not None:
                _, buffer = cv2.imencode('.jpg', frame_with_results)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    default_model_dir = ''
    default_model = 'yolov8n_full_integer_quant_edgetpu.tflite'
    default_labels = 'label.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--input', type=str, 
                        default='333 VID_20231011_170120.mp4')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host IP address for the Flask app')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port number for the Flask app')
    args = parser.parse_args()

    # Run video processing in a separate thread
    import threading
    video_thread = threading.Thread(target=video_stream, args=(args,), daemon=True)
    video_thread.start()

    # Start Flask server
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
