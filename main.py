import cv2
import numpy as np
import tensorflow.lite as tflite
from tflite_runtime.interpreter import Interpreter, load_delegate


# Load YOLO segmentation TFLite model
interpreter = tflite.Interpreter(model_path="best_float16.tflite", experimental_delegates=[load_delegate('libedgetpu.so.1')])
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

# Function to postprocess model output
def postprocess_output(frame, output):
    # Get the segmentation map
    segmentation_map = output[0]  # Assuming single output tensor
    segmentation_map_resized = cv2.resize(segmentation_map, (frame.shape[1], frame.shape[0]))
    # Normalize and convert to an 8-bit image for visualization
    colored_segmentation = (segmentation_map_resized * 255).astype(np.uint8)
    # Apply a color map to the segmentation map
    colored_segmentation = cv2.applyColorMap(colored_segmentation, cv2.COLORMAP_JET)
    # Overlay the segmentation map onto the original frame
    overlay = cv2.addWeighted(frame, 0.7, colored_segmentation, 0.3, 0)
    return overlay

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    preprocessed_frame = preprocess_frame(frame, input_shape)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Postprocess and display the output
    output_frame = postprocess_output(frame, output_data)
    cv2.imshow("YOLO Segmentation", output_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
