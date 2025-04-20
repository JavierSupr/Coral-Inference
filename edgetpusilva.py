import cv2
import time
import threading
import socket
import numpy as np
import struct
import json
import csv
from ultralytics import YOLO

# UDP Socket configuration
UDP_IP = "192.168.137.1"  # Replace with receiver's IP address
PORT_1 = 5010            # Port for first video stream
PORT_2 = 5011            # Port for second video stream
RESULTS_PORT_1 = 5012    # Port for inference results from Camera 1
RESULTS_PORT_2 = 5013    # Port for inference results from Camera 2
BUFFER_SIZE = 65000      # Max UDP packet size

# Initialize sockets for video streams and results
sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
results_sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
results_sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# CSV file for saving results
RESULTS_CSV = "inference_results.csv"

def process_segmentation(model_path, input_source, sock, video_port, results_sock, results_port, stream_name, imgsz=256, threshold=0.3, fps_target=30):
    """Runs YOLO segmentation on a camera stream with unique frame identifier."""
    
    model = YOLO(model=model_path, task="segment")
    cap = cv2.VideoCapture(input_source)

    fps_source = cap.get(cv2.CAP_PROP_FPS) or fps_target  
    frame_delay = 1.0 / fps_source  

    prev_time = time.time()
    frame_id = 0  # Unique identifier for each frame

    # Open CSV file for writing inference results
    with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        while cap.isOpened():
            start_time = time.time()
            frame_id += 1  # Increment frame ID on each loop
            
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(frame, conf=threshold, iou=0.2, imgsz=imgsz, verbose=False)
            
            objs_lst = []
            class_count = {}  # Dictionary to store class count
            
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
                        "frame_id": frame_id,  # Include frame_id in inference results
                        "camera": stream_name,
                        "id": int(obj_cls),
                        "label": label,
                        "conf": float(conf),
                        "bbox": bb.tolist(),
                        "seg": [s.tolist() for s in seg],
                    }
                    objs_lst.append(obj_data)

                    # Count the detected classes
                    if label in class_count:
                        class_count[label] += 1
                    else:
                        class_count[label] = 1

            # Prepare detection summary (e.g., 1 Truk Besar, 2 Truk Kecil)
            detection_summary = ", ".join([f"{count} {cls}" for cls, count in class_count.items()])

            fps = 1 / (time.time() - prev_time) if prev_time > 0 else 0
            prev_time = time.time()

            print(f"[INFO] {stream_name} - Frame ID: {frame_id} - FPS: {fps:.2f} - Detections: {detection_summary}")

            # Send inference results with frame_id
            inference_data = json.dumps({"frame_id": frame_id, "objects": objs_lst, "fps": fps})
            results_sock.sendto(inference_data.encode(), (UDP_IP, results_port))

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            # Split data into chunks
            chunks = [buffer[i:i + BUFFER_SIZE] for i in range(0, len(buffer), BUFFER_SIZE)]

            try:
                # Send frame_id first (as a 4-byte integer)
                sock.sendto(struct.pack("I", frame_id), (UDP_IP, video_port))  # Send frame ID
                
                # Send number of chunks
                sock.sendto(struct.pack("B", len(chunks)), (UDP_IP, video_port))

                # Send each chunk
                for chunk in chunks:
                    sock.sendto(chunk, (UDP_IP, video_port))

            except socket.error as e:
                print(f"[ERROR] Network issue in {stream_name}: {e}")
                break  

            elapsed_time = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed_time)  
            time.sleep(sleep_time)

            # Write to CSV after each frame
            writer.writerow([stream_name, frame_id, fps, detection_summary])

        cap.release()
        sock.close()
        print(f"[INFO] {stream_name} stream closed.")


def run_dual_camera_inference(model_path, cam1_source=0, cam2_source=1):
    """Runs YOLO segmentation on two cameras in parallel and streams video."""

    # Write headers to CSV file if not already present
    with open(RESULTS_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Camera", "Frame ID", "FPS", "Detected Objects"])

    thread1 = threading.Thread(target=process_segmentation, args=(
        model_path, cam1_source, sock1, PORT_1, results_sock1, RESULTS_PORT_1, "Camera 1"
    ))
    thread2 = threading.Thread(target=process_segmentation, args=(
        model_path, cam2_source, sock2, PORT_2, results_sock2, RESULTS_PORT_2, "Camera 2"
    ))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    print("Inference and streaming completed for both cameras.")

if __name__ == "__main__":
    YOLO_MODEL_PATH = "best_13-04-2025_full_integer_quant_edgetpu.tflite"
    cam1_source = "Camera 1 rev.mp4"
    cam2_source = "Camera 1 rev.mp4"
    run_dual_camera_inference(YOLO_MODEL_PATH, cam1_source, cam2_source)
