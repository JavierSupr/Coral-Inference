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
    input_image = input_image.astype(np.float32)
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

def main():
    # Initialize model
    model_path = 'best_full_integer_quant_edgetpu.tflite'  # Replace with your model path
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    input_size = (input_details[0]['shape'][2], input_details[0]['shape'][1])
    
    # Initialize cameras
    cap1, cap2 = initialize_cameras()
    
    # Load labels (adjust path as needed)
    with open('label.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
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
            detections1 = detect.get_objects(interpreter, score_threshold=0.4)
            
            # Process Camera 2
            input_tensor2 = preprocess_image(frame2, input_size)
            common.set_input(interpreter, input_tensor2)
            interpreter.invoke()
            detections2 = detect.get_objects(interpreter, score_threshold=0.4)
            
            # Print results for Camera 1
            print("\nCamera 1 Detections:")
            if detections1:
                for obj in detections1:
                    print(f"{labels[obj.id]}: {obj.score:.2f} at location "
                          f"({obj.bbox.xmin:.2f}, {obj.bbox.ymin:.2f}, "
                          f"{obj.bbox.xmax:.2f}, {obj.bbox.ymax:.2f})")
            else:
                print("No objects detected")
            
            # Print results for Camera 2
            print("\nCamera 2 Detections:")
            if detections2:
                for obj in detections2:
                    print(f"{labels[obj.id]}: {obj.score:.2f} at location "
                          f"({obj.bbox.xmin:.2f}, {obj.bbox.ymin:.2f}, "
                          f"{obj.bbox.xmax:.2f}, {obj.bbox.ymax:.2f})")
            else:
                print("No objects detected")
            
            # Calculate and print FPS
            fps = 1.0 / (time.time() - start_time)
            print(f"\nFPS: {fps:.2f}")
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        cap1.release()
        cap2.release()

if __name__ == '__main__':
    main()