import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the compiled TFLite model
try:
    interpreter = tflite.Interpreter(
        model_path="best_full_integer_quant_edgetpu.tflite",
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
    print("Edge TPU delegate loaded successfully.")
except ValueError as e:
    print("Failed to load Edge TPU delegate:", e)
    exit()

interpreter.allocate_tensors()

print("Loaded delegates:", interpreter._delegates)
print("Using Edge TPU:", "libedgetpu" in str(interpreter._delegates))

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model input shape
input_shape = input_details[0]['shape']  # Example: [1, 320, 320, 3]
print(f"Input shape: {input_shape}")

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

# Open two camera feeds or video files
cap1 = cv2.VideoCapture('video.mp4')  # First camera (use 0 or video file path)
cap2 = cv2.VideoCapture('333 VID_20231011_170120.mp4')  # Second camera (use 1 or video file path)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Unable to open one or both camera feeds.")
    exit()

# Measure FPS
prev_time = time.time()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Unable to read from one or both camera feeds.")
        break

    # Preprocess the frames
    input_data1 = preprocess_frame(frame1, input_shape)
    input_data2 = preprocess_frame(frame2, input_shape)

    # Process first frame
    interpreter.set_tensor(input_details[0]['index'], input_data1)
    start_time = time.time()
    interpreter.invoke()
    output_data1 = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    inference_time1 = time.time() - start_time

    # Process second frame
    interpreter.set_tensor(input_details[0]['index'], input_data2)
    start_time = time.time()
    interpreter.invoke()
    output_data2 = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    inference_time2 = time.time() - start_time

    # Postprocess results
    postprocess_output(output_data1)
    postprocess_output(output_data2)

    # Display the frames
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # Measure time and calculate FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time
    if elapsed_time > 0:
        fps = 1 / elapsed_time
        print(f"FPS: {fps:.2f}, Inference Time Camera 1: {inference_time1 * 1000:.2f} ms, Inference Time Camera 2: {inference_time2 * 1000:.2f} ms")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
