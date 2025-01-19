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


def draw_results(frame, results, labels):
    """
    Draw detection results on the frame.
    """
    for result in results:
        bbox = result["bbox"]
        label = result["label"]
        score = result["score"]

        # Convert normalized bbox to pixel values
        height, width, _ = frame.shape
        xmin, ymin, xmax, ymax = (
            int(bbox[0] * width),
            int(bbox[1] * height),
            int(bbox[2] * width),
            int(bbox[3] * height),
        )

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Draw label and score
        label_text = f"{label}: {score:.2f}"
        cv2.putText(
            frame,
            label_text,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return frame


def video_stream(args):
    global frame_with_results
    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    # Open video source
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))  # Use camera index
    else:
        cap = cv2.VideoCapture(args.input)  # Use video file

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to model input size
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

        # Perform inference
        run_inference(interpreter, cv2_im_rgb.tobytes())
        output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
        results = yolo_post_process(output_tensor, labels, args.threshold)

        # Draw results on the frame
        frame_with_results = draw_results(frame, results, labels)


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
    default_model = '240_yolov8n_full_integer_quant_edgetpu.tflite'
    default_labels = 'label.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--input', type=str, 
                        default='video.mp4')
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
