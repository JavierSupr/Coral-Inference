import cv2
import time
import threading
import socket
import numpy as np
import struct
import json
from ultralytics import YOLO
from datetime import datetime

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
    """Runs YOLO segmentation on a camera stream with dynamic frame delay to maintain correct FPS."""
    
    model = YOLO(model=model_path, task="segment")
    cap = cv2.VideoCapture(input_source)

    # Get FPS from video source
    #fps_source = cap.get(cv2.CAP_PROP_FPS) or fps_target  # Use target FPS if source FPS is unknown
    #frame_delay = 1.0 / fps_source  # Time delay per frame
    
    #prev_time = time.time()

    while cap.isOpened():
        #start_time = time.time()  # Start time of frame processing
        #frame_timestamp = datetime.now().strftime("%H:%M:%S.%f")  # Capture frame timestamp
        
        ret, frame = cap.read()
        if not ret:
            break  # End of stream
        
        # Run YOLO segmentation
        results = model.predict(frame, conf=threshold, imgsz=imgsz, verbose=False)
        #inference_timestamp = datetime.now().strftime("%H:%M:%S.%f")  # Capture inference timestamp

        objs_lst = []
        for out in results:
            masks = out.masks
            for index, box in enumerate(out.boxes):
                seg = masks.xy[index]
                obj_cls, conf, bb = (
                    int(box.cls.numpy()[0]),
                    float(box.conf.numpy()[0]),
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
                print(obj_data)
                objs_lst.append(obj_data)

        fps = 1 / (time.time() - prev_time) if prev_time > 0 else 0
        prev_time = time.time()

        # Print timestamps for comparison
        #print(f"\n[{stream_name}] Frame Captured: {frame_timestamp}")
        #print(f"[{stream_name}] Inference Completed: {inference_timestamp}")
        #print(f"[{stream_name}] Time Difference: {time.time() - start_time:.3f} seconds")

        # Send inference results with timestamps
        inference_data = json.dumps({
            "objects": objs_lst,
            "fps": fps,
        })
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

        # Ensure correct playback speed by sleeping for remaining time
        #elapsed_time = time.time() - start_time
        #sleep_time = max(0, frame_delay - elapsed_time)  # Avoid negative sleep time
        #time.sleep(sleep_time)

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
    cam2_source = "333 VID_20231011_170120_1.mp4"
    run_dual_camera_inference(YOLO_MODEL_PATH, cam1_source, cam2_source)
