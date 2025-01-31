import argparse
import cv2
import numpy as np
import time
from pycoral.adapters.common import input_size
from pycoral.adapters.segmentation import segment
from pycoral.utils.edgetpu import make_interpreter
from flask import Flask, Response

# Flask app initialization
app = Flask(__name__)

# Load TFLite model with Edge TPU
MODEL_PATH = "deeplabv3_mnv2_pascal_quant_edgetpu.tflite"
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

# Get input tensor shape
input_shape = input_size(interpreter)

# Function to preprocess frame
def preprocess_frame(frame, target_size):
    frame_resized = cv2.resize(frame, target_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = np.expand_dims(frame_rgb.astype(np.uint8), axis=0)
    return frame_normalized

# Function to overlay segmentation mask
def overlay_segmentation(frame, mask):
    colors = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
    mask_color = colors[mask]
    overlay = cv2.addWeighted(frame, 0.5, mask_color, 0.5, 0)
    return overlay

# Function to process video frame by frame
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        
        # Preprocess frame
        input_data = preprocess_frame(frame, input_shape)
        
        # Run inference
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()
        output_data = segment(interpreter)
        
        # Post-process output
        segmentation_mask = output_data.squeeze()
        overlayed_frame = overlay_segmentation(frame, segmentation_mask)
        
        # Print segmentation result
        unique_classes, counts = np.unique(segmentation_mask, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            print(f"Class {cls}: {count} pixels")
        
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.3f} seconds")

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', overlayed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Flask route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(args.video), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    args = parser.parse_args()
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
