import cv2
import time
import threading
import socket
import numpy as np
import struct
import json
from ultralytics import YOLO

# UDP Socket configuration
UDP_IP = "192.168.137.1"  # Replace with receiver's IP address
PORT_1 = 5005           # Port for first video stream
PORT_2 = 5006           # Port for second video stream
RESULTS_PORT = 5007     # Port for inference results
BUFFER_SIZE = 65000     # Max UDP packet size

# Initialize sockets for two video streams
sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
results_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Set frame size for both streams
WIDTH = 640
HEIGHT = 480

def process_segmentation(model_path, input_source, sock, port, stream_name, imgsz=256, threshold=0.4, fps_target=30):
    """Runs YOLO segmentation on a camera stream with dynamic frame skipping and sends video over UDP."""
    
    model = YOLO(model=model_path, task="segment")
    cap = cv2.VideoCapture(input_source)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    
    frame_skip = 1
    frame_count = 0
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of stream
        
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time
        desired_interval = 1.0 / fps_target
        
        if elapsed_time < desired_interval:
            frame_skip = max(1, frame_skip - 1)  # Reduce skipping if faster
        else:
            frame_skip = min(5, frame_skip + 1)  # Increase skipping if slower
        
        if frame_count % frame_skip == 0:
            prev_time = time.time()
            results = model.predict(frame, conf=threshold, imgsz=imgsz, verbose=False)
            
            objs_lst = []
            for out in results:
                masks = out.masks
                for index, box in enumerate(out.boxes):
                    seg = masks.xy[index]
                    obj_cls, conf, bb = (
                        box.cls.numpy()[0],
                        box.conf.numpy()[0],
                        box.xyxy.numpy()[0],
                    )
                    label = out.names[int(obj_cls)]
                    obj_data = {
                        "id": int(obj_cls),
                        "label": label,
                        "conf": float(conf),
                        "bbox": bb.tolist(),
                        "seg": [s.tolist() for s in seg],
                    }
                    objs_lst.append(obj_data)
                    print(obj_data)
            
            fps = 1 / elapsed_time if elapsed_time > 0 else 0
            inference_data = json.dumps({"objects": objs_lst, "fps": fps})
            results_sock.sendto(inference_data.encode(), (UDP_IP, RESULTS_PORT))
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            # Split data into chunks (UDP max packet size is ~65507 bytes)
            chunks = [buffer[i:i + BUFFER_SIZE] for i in range(0, len(buffer), BUFFER_SIZE)]
            
            try:
                # Send number of chunks
                sock.sendto(struct.pack("B", len(chunks)), (UDP_IP, port))
                
                # Send each chunk
                for chunk in chunks:
                    sock.sendto(chunk, (UDP_IP, port))
            
            except socket.error as e:
                print(f"[ERROR] Network issue in {stream_name}: {e}")
                break  # Stop sending if network fails
    
    cap.release()
    sock.close()
    print(f"[INFO] {stream_name} stream closed.")


def run_dual_camera_inference(model_path, cam1_source=0, cam2_source=1):
    """Runs YOLO segmentation on two cameras in parallel and streams video."""
    
    thread1 = threading.Thread(target=process_segmentation, args=(model_path, cam1_source, sock1, PORT_1, "Camera 1"))
    thread2 = threading.Thread(target=process_segmentation, args=(model_path, cam2_source, sock2, PORT_2, "Camera 2"))
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    print("Inference and streaming completed for both cameras.")

if __name__ == "__main__":
    YOLO_MODEL_PATH = "best_full_integer_quant_edgetpu.tflite"
    cam1_source = "333 VID_20231011_170120_1.mp4"
    cam2_source = "333-vid-20231011-170120_Tt2GmTrq.mp4"
    run_dual_camera_inference(YOLO_MODEL_PATH, cam1_source, cam2_source)
