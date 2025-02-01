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

        #processed_frame = preprocess_frame(frame)
        #if processed_frame is None:
        #    continue

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], frame)
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