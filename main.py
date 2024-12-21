import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the compiled TFLite model
interpreter = tflite.Interpreter(
    model_path="best_full_integer_quant_edgetpu.tflite",
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model input shape
input_shape = input_details[0]['shape']  # Example: [1, 320, 320, 3]

# Preprocess each video frame
def preprocess_frame(frame, input_shape):
    image = cv2.resize(frame, (input_shape[1], input_shape[2]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize to [-128, 127] for INT8
    normalized_image = image.astype(np.float32) - 128
    input_data = np.expand_dims(normalized_image, axis=0).astype(np.int8)
    return input_data

# Postprocess the model output to print detected labels and confidences
def postprocess_output(output_data, threshold=0.5):
    """
    Postprocess YOLOv8 segmentation output.
    - `output_data[0]` contains bounding boxes and class scores.
    """
    boxes = output_data[0]  # Bounding boxes, confidences, and class indices
    for box in boxes:
        # Ensure box is a 1D array
        if len(box.shape) == 1:
            confidence = box[4]
            if confidence > threshold:  # Confidence threshold
                class_id = int(box[5])
                print(f"Class ID: {class_id}, Confidence: {confidence:.2f}")

# Open video file or camera feed
cap = cv2.VideoCapture("333 VID_20231011_170120.mp4")  # Use 0 for webcam or replace with video file path
print("Loaded delegates:", interpreter._delegates)
print("Using Edge TPU:", "libedgetpu" in str(interpreter._delegates))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = preprocess_frame(frame, input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get model outputs
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    # Postprocess and print results
    postprocess_output(output_data)

    # Break loop with 'q' key
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
cv2.destroyAllWindows()
