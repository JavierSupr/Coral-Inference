import cv2
import numpy as np
import time
import threading
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size

# Define class names for segmentation labels
CLASS_NAMES = {
    0: "Background", 1: "Aeroplane", 2: "Bicycle", 3: "Bird", 4: "Boat",
    5: "Bottle", 6: "Bus", 7: "Car", 8: "Cat", 9: "Chair", 10: "Cow",
    11: "Dining table", 12: "Dog", 13: "Horse", 14: "Motorbike", 15: "Person",
    16: "Potted plant", 17: "Sheep", 18: "Sofa", 19: "Train", 20: "TV/Monitor"
}

def preprocess_frame(frame, input_size=(128, 128)):
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
    
    return np.expand_dims(canvas, axis=0)  # Keep raw uint8

def process_video(video_path, model_path, video_id):
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
        processed_frame = preprocess_frame(frame)
        if processed_frame is None:
            continue
        
        interpreter.set_tensor(input_details[0]['index'], processed_frame)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        segmentation_mask = output_data[0]  # Extract mask from batch
        
        unique_classes, counts = np.unique(segmentation_mask, return_counts=True)
        total_pixels = np.sum(counts)
        
        detected_classes = []
        for cls, count in zip(unique_classes, counts):
            class_name = CLASS_NAMES.get(cls, "Unknown")
            confidence = count / total_pixels
            detected_classes.append((class_name, confidence))
            print(f"[Video {video_id}] Class: {class_name}, Confidence: {confidence:.2f}")
        
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        fps = 1.0 / frame_time
        print(f"[Video {video_id}] Real-time FPS: {fps:.2f}")
        
        frame_count += 1
    
    cap.release()
    end_time = time.time()
    avg_fps = frame_count / (end_time - start_time)
    print(f"[Video {video_id}] Processed {frame_count} frames at an average of {avg_fps:.2f} FPS")

if __name__ == "__main__":
    model_path = "keras_post_training_unet_mv2_128_quant_edgetpu.tflite"
    video_paths = [
        "333-vid-20231011-170120_Tt2GmTrq.mp4",  # Replace with actual path
        "WIN_20240924_12_35_51_Pro.mp4"   # Replace with actual path
    ]
    
    threads = []
    for i, video_path in enumerate(video_paths):
        thread = threading.Thread(target=process_video, args=(video_path, model_path, i+1))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print("Processing of both videos is complete.")
