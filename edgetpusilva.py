import cv2
import time
import threading
import socket
import numpy as np
import struct
import json
from ultralytics import YOLO
import queue

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

def inference_worker(model, frame_queue, results_queue, threshold=0.4, imgsz=256):
    """Background thread that runs YOLO inference asynchronously."""
    while True:
        if not frame_queue.empty():
            frame, timestamp = frame_queue.get()
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
                    print(bb)
                    label = out.names[int(obj_cls)]
                    obj_data = {
                        "id": int(obj_cls),
                        "label": label,
                        "conf": float(conf),
                        "bbox": bb.tolist(),
                        "seg": [s.tolist() for s in seg],
                    }
                    objs_lst.append(obj_data)
            
            # Store inference results with timestamp
            results_queue.put({"objects": objs_lst, "timestamp": timestamp})

def process_segmentation(model_path, input_source, sock, port, stream_name, imgsz=256, threshold=0.4, fps_target=30):
    """Runs YOLO segmentation asynchronously while streaming video over UDP."""
    
    model = YOLO(model=model_path, task="segment")
    cap = cv2.VideoCapture(input_source)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    fps_source = cap.get(cv2.CAP_PROP_FPS) or fps_target  # Use target FPS if source FPS is unknown
    frame_delay = 1.0 / fps_source  # Time delay per frame
    
    frame_queue = queue.Queue(maxsize=5)
    results_queue = queue.Queue(maxsize=5)


    # Start inference thread
    inference_thread = threading.Thread(target=inference_worker, args=(model, frame_queue, results_queue, threshold, imgsz))
    inference_thread.daemon = True
    inference_thread.start()

    prev_time = time.time()

    while cap.isOpened():
        start_time = time.time()  # Track start time for frame processing
        
        ret, frame = cap.read()
        if not ret:
            break  # End of stream
        
        # Send frame to inference queue
        if frame_queue.qsize() < 5:  # Prevent overloading
            frame_queue.put((frame.copy(), time.time()))

        # Fetch inference results if available
        if not results_queue.empty():
            inference_data = results_queue.get()
            inference_data["fps"] = 1 / (time.time() - prev_time) if prev_time > 0 else 0
            prev_time = time.time()

            # Send inference data over UDP
            results_sock.sendto(json.dumps(inference_data).encode(), (UDP_IP, RESULTS_PORT))

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
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed_time)  # Avoid negative sleep time
        time.sleep(sleep_time)

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
    cam1_source = "333-vid-20231011-170120_Tt2GmTrq.mp4"
    cam2_source = "333-vid-20231011-170120_Tt2GmTrq.mp4"
    run_dual_camera_inference(YOLO_MODEL_PATH, cam1_source, cam2_source)
