import cv2
import numpy as np
import time
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect

def initialize_cameras():
    # Initialize two cameras (adjust device IDs as needed)
    cap1 = cv2.VideoCapture("333 VID_20231011_170120.mp4")
    cap2 = cv2.VideoCapture("video.mp4")
    
    # Set resolution for both cameras (adjust as needed)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap1, cap2

def preprocess_image(image, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, input_size)
    input_image = input_image.astype(np.float32) / 255.0  # Normalize to [0,1]
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

def process_output(interpreter, score_threshold=0.4):
    """Process YOLO model output"""
    output_details = interpreter.get_output_details()
    
    # Print output tensor details for debugging
    print("\nOutput Tensor Details:")
    for i, output in enumerate(output_details):
        print(f"Output {i}: shape={interpreter.get_tensor(output['index']).shape}")
    
    # Get outputs (adjust indices based on your model's output format)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Print first detection for debugging
    if output_data.size > 0:
        print(f"\nFirst detection raw data: {output_data[0][0]}")
    
    # Parse predictions
    detections = []
    for detection in output_data[0]:
        # YOLO output format: [x, y, w, h, confidence, class_id, class_scores...]
        confidence = float(detection[4])
        
        if confidence >= score_threshold:
            class_id = int(detection[5])
            xmin = float(detection[0] - (detection[2] / 2))  # center_x - width/2
            ymin = float(detection[1] - (detection[3] / 2))  # center_y - height/2
            xmax = float(detection[0] + (detection[2] / 2))  # center_x + width/2
            ymax = float(detection[1] + (detection[3] / 2))  # center_y + height/2
            
            detections.append({
                'class_id': class_id,
                'confidence': confidence,
                'bbox': [xmin, ymin, xmax, ymax]
            })
    
    return detections

def print_detection(detection, labels, camera_id):
    """Safely print detection results with bounds checking"""
    try:
        class_id = detection['class_id']
        if class_id < 0 or class_id >= len(labels):
            print(f"Warning: Invalid class_id {class_id} (labels length: {len(labels)})")
            label = f"Unknown_{class_id}"
        else:
            label = labels[class_id]
            
        print(f"{label}: {detection['confidence']:.2f} at location "
              f"({detection['bbox'][0]:.2f}, {detection['bbox'][1]:.2f}, "
              f"{detection['bbox'][2]:.2f}, {detection['bbox'][3]:.2f})")
    except Exception as e:
        print(f"Error printing detection for camera {camera_id}: {str(e)}")
        print(f"Raw detection data: {detection}")

def main():
    # Initialize model
    model_path = 'best_full_integer_quant_edgetpu.tflite'  # Replace with your model path
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    
    # Get and print model details
    input_details = interpreter.get_input_details()
    print("\nModel Input Details:")
    print(f"Input shape: {input_details[0]['shape']}")
    input_size = (input_details[0]['shape'][2], input_details[0]['shape'][1])
    
    # Initialize cameras
    cap1, cap2 = initialize_cameras()
    
    # Load and print labels
    try:
        with open('label.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"\nLoaded {len(labels)} labels")
    except Exception as e:
        print(f"Error loading labels: {str(e)}")
        labels = []
    
    try:
        while True:
            start_time = time.time()
            
            # Read frames from both cameras
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print("Error reading from cameras")
                break
            
            # Process Camera 1
            input_tensor1 = preprocess_image(frame1, input_size)
            common.set_input(interpreter, input_tensor1)
            interpreter.invoke()
            detections1 = process_output(interpreter)
            
            # Process Camera 2
            input_tensor2 = preprocess_image(frame2, input_size)
            common.set_input(interpreter, input_tensor2)
            interpreter.invoke()
            detections2 = process_output(interpreter)
            
            # Print results for Camera 1
            print("\nCamera 1 Detections:")
            if detections1:
                for det in detections1:
                    print_detection(det, labels, 1)
            else:
                print("No objects detected")
            
            # Print results for Camera 2
            print("\nCamera 2 Detections:")
            if detections2:
                for det in detections2:
                    print_detection(det, labels, 2)
            else:
                print("No objects detected")
            
            # Calculate and print FPS
            fps = 1.0 / (time.time() - start_time)
            print(f"\nFPS: {fps:.2f}")

                
    finally:
        # Clean up
        cap1.release()
        cap2.release()

if __name__ == '__main__':
    main()