import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Path to your YOLO tflite model
MODEL_PATH = "best_full_integer_quant_edgetpu.tflite"
LABELS_PATH = "label.txt"

# Load labels
def load_labels(path):
    with open(path, "r") as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = load_labels(LABELS_PATH)

# Initialize the Edge TPU interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH, experimental_delegates=[
    tflite.load_delegate('libedgetpu.so.1')
])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input dimensions
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

# Initialize cameras
camera_1 = cv2.VideoCapture("333 VID_20231011_170120.mp4")  # Adjust device index if needed
camera_2 = cv2.VideoCapture("video.mp4")

if not camera_1.isOpened() or not camera_2.isOpened():
    print("Error: Unable to open cameras.")
    exit()

print("Starting inference on two cameras...")

try:
    while True:
        for cam_index, camera in enumerate([camera_1, camera_2], start=1):
            ret, frame = camera.read()
            if not ret:
                print(f"Error: Unable to read from Camera {cam_index}.")
                continue

            # Preprocess frame
            resized_frame = cv2.resize(frame, (width, height))
            input_data = np.expand_dims(resized_frame, axis=0)

            # Check and adjust input data type
            if input_details[0]['dtype'] == np.int8:
                # Scale input to INT8 range (-128 to 127)
                input_data = (input_data.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
                input_data = (input_data * 127).astype(np.int8)  # Scale to INT8
            elif input_details[0]['dtype'] == np.uint8:
                # No need to change, already UINT8
                input_data = input_data.astype(np.uint8)
            else:
                raise ValueError(f"Unsupported input data type: {input_details[0]['dtype']}")

            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Retrieve and process results
            output_data = interpreter.get_tensor(output_details[0]['index'])
            detections = output_data[0]  # Assuming one output tensor with boxes

            print(f"Camera {cam_index} Results:")
            for detection in detections:
                score = detection[1]
                if score > 0.5:  # Threshold for detection confidence
                    label_index = int(detection[0])
                    bbox = detection[2:]
                    try:
                        label_name = labels[label_index]
                    except KeyError:
                        label_name = f"Unknown ({label_index})"
                    print(f"  - {label_name}: {score*100:.2f}% ({bbox})")


except KeyboardInterrupt:
    print("Inference stopped.")
finally:
    camera_1.release()
    camera_2.release()
    print("Cameras released.")
