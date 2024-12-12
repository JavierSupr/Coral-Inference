import cv2
import numpy as np
import tensorflow.lite as tflite

# Load YOLO segmentation TFLite model
interpreter = tflite.Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess frame for the model
def preprocess_frame(frame, input_shape):
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0).astype(np.float32)

# Function to postprocess model output
def postprocess_output(frame, output):
    segmentation_map = cv2.resize(output[0], (frame.shape[1], frame.shape[0]))
    colored_segmentation = (segmentation_map * 255).astype(np.uint8)
    overlay = cv2.addWeighted(frame, 0.7, cv2.applyColorMap(colored_segmentation, cv2.COLORMAP_JET), 0.3, 0)
    return overlay

# Open video capture
video_path = "333 VID_20231011_170120.mp4"  # Replace with 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file or webcam.")
    exit()

input_shape = input_details[0]['shape']

# Video writer for saving output
output_video = cv2.VideoWriter(
    'output.avi', 
    cv2.VideoWriter_fourcc(*'XVID'), 
    30, 
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    preprocessed_frame = preprocess_frame(frame, input_shape)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_frame = postprocess_output(frame, output_data)
    output_video.write(output_frame)

cap.release()
output_video.release()
