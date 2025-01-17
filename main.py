import time
import numpy as np
import cv2
from pycoral.adapters.common import input_tensor, output_tensor
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter

# Load the model
MODEL_PATH = '240_yolov8n_full_integer_quant_edgetpu.tflite'
THRESHOLD = 0.5
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

# Initialize cameras
CAMERA_1_ID = "video.mp4"  # First camera (e.g., /dev/video0)
CAMERA_2_ID = "333 VID_20231011_170120.mp4"  # Second camera (e.g., /dev/video1)
cap1 = cv2.VideoCapture(CAMERA_1_ID)
cap2 = cv2.VideoCapture(CAMERA_2_ID)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Unable to access one or both cameras.")
    exit(1)

def preprocess_frame(frame, size):
    """Resize and preprocess the frame to feed into the model."""
    resized = cv2.resize(frame, size)
    # Convert BGR to RGB (if needed by the model)
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to [0, 1] range
    normalized_frame = rgb_frame / 255.0
    # Ensure the array has the correct shape
    return np.expand_dims(normalized_frame.astype('float32'), axis=0)


def run_inference(interpreter, frame):
    """Run inference on a single frame."""
    size = input_tensor(interpreter).shape[1:3]
    preprocessed = preprocess_frame(frame, size)

    # Load the tensor and invoke the interpreter
    input_tensor(interpreter)[:, :] = preprocessed
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Get detection results
    detections = get_objects(interpreter, THRESHOLD)
    return detections, inference_time

def print_detections(detections, cam_id, inference_time, fps):
    """Print the detection results to the CLI."""
    print(f"Camera {cam_id} Detections:")
    print(f"  FPS: {fps:.2f}, Inference Time: {inference_time * 1000:.2f} ms")
    for detection in detections:
        print(f"  ID: {detection.id}, Score: {detection.score:.2f}, Bounding Box: {detection.bbox}")

try:
    frame_count = 0
    start_time = time.time()

    while True:
        # Read frames from both cameras
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Error: Unable to read frames from one or both cameras.")
            break

        frame_count += 1

        # Run inference
        detections1, inference_time1 = run_inference(interpreter, frame1)
        detections2, inference_time2 = run_inference(interpreter, frame2)

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Print results
        print_detections(detections1, cam_id=1, inference_time=inference_time1, fps=fps)
        print_detections(detections2, cam_id=2, inference_time=inference_time2, fps=fps)

        time.sleep(0.1)  # Add a small delay to limit output frequency

except KeyboardInterrupt:
    print("Stopping inference.")

finally:
    # Release resources
    cap1.release()
    cap2.release()
    print("Cameras released.")
