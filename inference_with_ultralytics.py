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

def main():
    model_path = "deeplabv3_mnv2_pascal_quant_edgetpu.tflite"  # Change to correct model path
    video_path = "333-vid-20231011-170120_Tt2GmTrq.mp4"

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

        frame_start_time = time.time()

        # Ensure frame has correct dimensions (1, 513, 513, 3)
        input_tensor = np.expand_dims(frame, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        # Get segmentation output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        segmentation_mask = output_data[0]  # Extract mask from batch
        
        # Calculate class confidence based on frequency in segmentation mask
        unique_classes, counts = np.unique(segmentation_mask, return_counts=True)
        total_pixels = np.sum(counts)
        
        detected_classes = []
        for cls, count in zip(unique_classes, counts):
            class_name = CLASS_NAMES.get(cls, "Unknown")
            confidence = count / total_pixels  # Approximate confidence as pixel proportion
            detected_classes.append((class_name, confidence))
            print(f"Class: {class_name}, Confidence: {confidence:.2f}")
        
        # Overlay segmentation mask on frame
        frame[segmentation_mask > 0] = [0, 255, 0]  # Apply green mask
        
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        fps = 1.0 / frame_time
        print(f"Real-time FPS: {fps:.2f}")
        
        frame_count += 1
    
    cap.release()
    end_time = time.time()
    avg_fps = frame_count / (end_time - start_time)
    print(f"Processed {frame_count} frames at an average of {avg_fps:.2f} FPS")

if __name__ == "__main__":
    main()
