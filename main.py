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
    # Resize frame to the input size required by the model
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
    # Add a batch dimension and convert to the required dtype
    return np.expand_dims(normalized_frame, axis=0).astype(input_details[0]['dtype'])

# Open video capture (can be a file path or a camera index)
video_path = "333 VID_20231011_170120.mp4"  # Replace with 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file or webcam.")
    exit()

# Retrieve input shape from the model
input_shape = input_details[0]['shape']

# Ensure input shape has the correct number of dimensions
if len(input_shape) != 4 or input_shape[0] != 1:
    print("Error: Model input shape must be [1, height, width, channels].")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Preprocess frame
    preprocessed_frame = preprocess_frame(frame, input_shape)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Display inference result on CLI
    print(f"Frame {frame_count}:")
    print(output_data)

    # Break on 'q' key press (optional, for interrupting long processing)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
print("Inference completed.")
