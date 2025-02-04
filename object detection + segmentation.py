import cv2
import numpy as np
import time
import threading
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter

# Load models
DETECTION_MODEL = "ssd_mobilenet_v1_coco_quant_postprocess_edgetpu (1).tflite"
SEGMENTATION_MODEL = "deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite"

detection_interpreter = make_interpreter(DETECTION_MODEL)
detection_interpreter.allocate_tensors()

#segmentation_interpreter = make_interpreter(SEGMENTATION_MODEL)
#segmentation_interpreter.allocate_tensors()

def detect_vehicle(frame):
    """Runs the vehicle detection model on a frame."""
    resized = cv2.resize(frame, input_size(detection_interpreter))
    input_tensor = np.expand_dims(resized, axis=0)
    
    detection_interpreter.set_tensor(detection_interpreter.get_input_details()[0]['index'], input_tensor)
    detection_interpreter.invoke()
    
    objs = get_objects(detection_interpreter, 0.4)  # Detection threshold
    return objs

#def run_segmentation(frame):
#    """Runs the segmentation model on the detected truck in a separate thread."""
#    resized = cv2.resize(frame, input_size(segmentation_interpreter))
#    input_tensor = np.expand_dims(resized, axis=0)
#    
#    segmentation_interpreter.set_tensor(segmentation_interpreter.get_input_details()[0]['index'], input_tensor)
#    segmentation_interpreter.invoke()
#    
#    segmentation_output = segmentation_interpreter.tensor(segmentation_interpreter.get_output_details()[0]['index'])()
#    cv2.imwrite("segmented_truck.png", segmentation_output)
#    print("Truck detected and segmented.")

def process_camera(cap, camera_id):
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        print(f"Camera {camera_id} - FPS: {fps:.2f}")
        
        detections = detect_vehicle(frame)
        for obj in detections:
            bbox = obj.bbox
            label = obj.id  # Assuming label ID corresponds to vehicle type
            
            if label == 7:  # Assuming label 7 is 'truck'
                x1, y1, x2, y2 = bbox
                print(f"Camera {camera_id} - Truck detected at ({x1}, {y1}), ({x2}, {y2})")
                truck_crop = frame[y1:y2, x1:x2]
                
                #segmentation_thread = threading.Thread(target=run_segmentation, args=(truck_crop,))
                #segmentation_thread.start()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

def main():
    cap1 = cv2.VideoCapture("333-vid-20231011-170120_Tt2GmTrq.mp4")  # First camera
    cap2 = cv2.VideoCapture("WIN_20240924_12_35_51_Pro.mp4")  # Second camera
    
    thread1 = threading.Thread(target=process_camera, args=(cap1, 1))
    thread2 = threading.Thread(target=process_camera, args=(cap2, 2))
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    main()
