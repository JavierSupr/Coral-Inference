import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the compiled TFLite model
try:
    interpreter = tflite.Interpreter(
        model_path="192_yolov8n-seg_full_integer_quant_edgetpu.tflite",
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
    print("Edge TPU delegate loaded successfully.")
except ValueError as e:
    print("Failed to load Edge TPU delegate:", e)

interpreter.allocate_tensors()

print("Loaded delegates:", interpreter._delegates)
print("Using Edge TPU:", "libedgetpu" in str(interpreter._delegates))

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model input shape
input_shape = input_details[0]['shape']  # Example: [1, 320, 320, 3]
print(f" input shape {input_shape}")

# Preprocess each video frame
def preprocess_frame(frame, input_shape):
    image = cv2.resize(frame, (input_shape[1], input_shape[2]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalized_image = (image - 127.5) / 127.5  # Normalize for int8 quantization
    input_data = np.expand_dims(normalized_image, axis=0).astype(np.int8)
    return input_data

# Postprocess the model output to print detected labels and confidences
def postprocess_output(output_data, threshold=0.3):
    boxes = output_data[0]  # Bounding boxes, confidences, and class indices
    for box in boxes:
        if len(box.shape) == 1:
            confidence = box[4]
            if confidence > threshold:  # Confidence threshold
                class_id = int(box[5])
                print(f"Class ID: {class_id}, Confidence: {confidence:.2f}")

# Open video file or camera feed
cap = cv2.VideoCapture("video.mp4")  # Use 0 for webcam or replace with video file path

# Measure FPS
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = preprocess_frame(frame, input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Measure inference time
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Get model outputs
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    # Postprocess the results
    postprocess_output(output_data)

    # Measure time and calculate FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time
    if elapsed_time > 0:
        fps = 1 / elapsed_time
        print(f"FPS: {fps:.2f}, Inference Time: {inference_time * 1000:.2f} ms")

cap.release()
