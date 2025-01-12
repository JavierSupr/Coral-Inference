import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

try:
    interpreter = tflite.Interpreter(
        model_path="best_full_integer_quant_edgetpu.tflite",
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
print(f"Input shape: {input_shape}")

# Preprocess each video frame
def preprocess_frame(frame, input_shape):
    image = cv2.resize(frame, (input_shape[1], input_shape[2]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Print min/max values before normalization
    print(f"Frame min/max before normalization: {np.min(image)}, {np.max(image)}")
    
    normalized_image = (image - 127.5) / 127.5
    input_data = np.expand_dims(normalized_image, axis=0).astype(np.int8)
    
    # Print min/max values after normalization
    print(f"Input data min/max after normalization: {np.min(input_data)}, {np.max(input_data)}")
    return input_data

# Postprocess the model output to print detected labels and confidences
def postprocess_output(output_data, threshold=0.3):
    print("Output data shapes:", [output.shape for output in output_data])
    print("Output data types:", [output.dtype for output in output_data])
    
    boxes = output_data[0]
    print(f"First few values in output: {boxes[0][:10]}")  # Print first 10 values
    
    if len(boxes.shape) == 2:  # If output is in expected format [num_boxes, box_data]
        for i, box in enumerate(boxes):
            confidence = box[4] if len(box) > 4 else 0
            if confidence > threshold:
                class_id = int(box[5]) if len(box) > 5 else -1
                x1, y1, x2, y2 = box[0:4]
                print(f"Box {i}: coords({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), "
                      f"confidence: {confidence:.2f}, class_id: {class_id}")
    else:
        print(f"Unexpected output shape: {boxes.shape}")

# Open video streams for two cameras
cap1 = cv2.VideoCapture('video.mp4')  # First camera
cap2 = cv2.VideoCapture('333 VID_20231011_170120.mp4')  # Second camera

print("\nModel Details:")
print("Input Details:", input_details)
print("Output Details:", output_details)
# Measure FPS
prev_time1 = time.time()
prev_time2 = time.time()

while cap1.isOpened() and cap2.isOpened():
    # Read frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Failed to capture frames from one or both cameras.")
        break

    # Preprocess the frames
    input_data1 = preprocess_frame(frame1, input_shape)
    input_data2 = preprocess_frame(frame2, input_shape)

    # Inference for the first camera
    interpreter.set_tensor(input_details[0]['index'], input_data1)
    start_time1 = time.time()
    interpreter.invoke()
    inference_time1 = time.time() - start_time1
    output_data1 = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    print("Camera 1 Results:")
    postprocess_output(output_data1)

    # Measure time and calculate FPS for the first camera
    current_time1 = time.time()
    elapsed_time1 = current_time1 - prev_time1
    prev_time1 = current_time1
    if elapsed_time1 > 0:
        fps1 = 1 / elapsed_time1
        print(f"Camera 1 - FPS: {fps1:.2f}, Inference Time: {inference_time1 * 1000:.2f} ms")

    # Inference for the second camera
    interpreter.set_tensor(input_details[0]['index'], input_data2)
    start_time2 = time.time()
    interpreter.invoke()
    inference_time2 = time.time() - start_time2
    output_data2 = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    print("Camera 2 Results:")
    postprocess_output(output_data2)

    # Measure time and calculate FPS for the second camera
    current_time2 = time.time()
    elapsed_time2 = current_time2 - prev_time2
    prev_time2 = current_time2
    if elapsed_time2 > 0:
        fps2 = 1 / elapsed_time2
        print(f"Camera 2 - FPS: {fps2:.2f}, Inference Time: {inference_time2 * 1000:.2f} ms")

cap1.release()
cap2.release()
cv2.destroyAllWindows()
