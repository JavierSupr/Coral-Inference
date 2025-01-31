import cv2
import numpy as np
import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size

# Define class names for segmentation labels
CLASS_NAMES = {
    0: "Background",
    1: "Aeroplane",
    2: "Bicycle",
    3: "Bird",
    4: "Boat",
    5: "Bottle",
    6: "Bus",
    7: "Car",
    8: "Cat",
    9: "Chair",
    10: "Cow",
    11: "Dining table",
    12: "Dog",
    13: "Horse",
    14: "Motorbike",
    15: "Person",
    16: "Potted plant",
    17: "Sheep",
    18: "Sofa",
    19: "Train",
    20: "TV/Monitor"
}

def preprocess_frame(frame, input_size=(513, 513)):
    if frame is None or frame.size == 0:
        return None
    
    height, width = frame.shape[:2]
    scale = min(input_size[0] / width, input_size[1] / height)
    new_width = min(int(width * scale), input_size[0])
    new_height = min(int(height * scale), input_size[1])
    
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    
    y_offset = max(0, (input_size[1] - new_height) // 2)
    x_offset = max(0, (input_size[0] - new_width) // 2)
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    # Convert to UINT8 (Edge TPU model requires UINT8)
    return np.expand_dims(canvas, axis=0)  # No need for normalization, keep raw uint8

def main():
    model_path = "mobilenetv2_deeplabv3_edgetpu.tflite"  # Change to correct model path
    video_path = "333 VID_20231011_170120.mp4"

    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        if processed_frame is None:
            continue

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], processed_frame)
        interpreter.invoke()

        # Get segmentation output (usually a mask)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        segmentation_mask = output_data[0]  # Extract mask from batch
        
        # Overlay segmentation mask on frame
        mask_resized = cv2.resize(segmentation_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        frame[mask_resized > 0] = [0, 255, 0]  # Apply green mask
        
        # Print detected classes
        unique_classes = np.unique(mask_resized)
        detected_classes = [CLASS_NAMES.get(cls, "Unknown") for cls in unique_classes]
        print(f"Detected Classes: {detected_classes}")
        
        frame_count += 1
    
    cap.release()
    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    print(f"Processed {frame_count} frames at {fps:.2f} FPS")

if __name__ == "__main__":
    main()
