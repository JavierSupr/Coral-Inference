import cv2
import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

# Load the model
model_path = "best_full_integer_quant_edgetpu.tflite"
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
input_shape = common.input_size(interpreter)
print(f"Model input shape: {input_shape}")

# Open cameras
cap1 = cv2.VideoCapture('video.mp4')  # First camera
cap2 = cv2.VideoCapture('333 VID_20231011_170120.mp4')  # Second camera

# Preprocess frame function
def preprocess_frame(frame, input_shape):
    resized_frame = cv2.resize(frame, input_shape)
    return resized_frame

# Process detections
def process_detections(detections, frame, color):
    for obj in detections:
        bbox = obj.bbox
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), color, 2)
        label = f"ID: {obj.id}, Conf: {obj.score:.2f}"
        cv2.putText(frame, label, (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Main loop
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Preprocess frames
    input_data1 = preprocess_frame(frame1, input_shape)
    input_data2 = preprocess_frame(frame2, input_shape)

    # Run inference on first frame
    common.set_input(interpreter, input_data1)
    interpreter.invoke()
    detections1 = detect.get_objects(interpreter, 0.3)

    # Run inference on second frame
    common.set_input(interpreter, input_data2)
    interpreter.invoke()
    detections2 = detect.get_objects(interpreter, 0.3)

    # Draw detections on frames
    process_detections(detections1, frame1, (0, 255, 0))
    process_detections(detections2, frame2, (255, 0, 0))

    # Show frames
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
