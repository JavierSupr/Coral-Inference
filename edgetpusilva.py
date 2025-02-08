import time
import socket
import json
import cv2
import numpy as np
from ultralytics import YOLO

# Define UDP settings
UDP_IP = "192.168.137.1"  # Replace with your PC's IP address
VIDEO_PORT = 5004  # Video stream via RTP
RESULTS_PORT = 5005  # Inference results via UDP

# Create UDP sockets
video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Video stream
results_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Inference data

def process_segmentation(model_path, input_path, imgsz=256, threshold=0.4, verbose=True, show=False, classes=None):
    """Runs YOLO segmentation and sends both video + inference results over UDP."""
    
    # Load the YOLO model
    model = YOLO(model=model_path, task="segment")

    # Open video file or webcam
    cap = cv2.VideoCapture(input_path)  

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Run YOLO segmentation
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

        # Compute FPS
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        # Send inference results via UDP (JSON format)
        inference_data = json.dumps({"objects": objs_lst, "fps": fps})
        results_sock.sendto(inference_data.encode(), (UDP_IP, RESULTS_PORT))

        # Encode and send video frame via UDP (RTP)
        _, buffer = cv2.imencode(".jpg", frame)
        video_sock.sendto(buffer, (UDP_IP, VIDEO_PORT))

        # Show locally (optional)
        if show:
            cv2.imshow("Streaming", frame)
            if cv2.waitKey(1) == 27:  # Press ESC to exit
                break

    cap.release()
    cv2.destroyAllWindows()

# Define paths and parameters
model_path = "best_full_integer_quant_edgetpu.tflite"
input_path = "333 VID_20231011_170120_1.mp4"
imgsz = 256
threshold = 0.4
verbose = True
show = False

# Run segmentation and stream results
process_segmentation(model_path, input_path, imgsz, threshold, verbose, show)
