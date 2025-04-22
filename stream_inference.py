import cv2
import time
import threading
import socket
import numpy as np
import struct
import csv
from ultralytics import YOLO

# UDP configuration
PORT_1 = 5015
RESULTS_CSV = "inference_results.csv"
BUFFER_SIZE = 65000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT_1))
sock.settimeout(1.0)

def receive_udp_stream():
    try:
        # Abaikan frame_id
        sock.recvfrom(4)
        
        # Terima jumlah potongan
        num_chunks_data, _ = sock.recvfrom(1)
        num_chunks = struct.unpack("B", num_chunks_data)[0]

        # Terima semua potongan
        chunks = []
        for _ in range(num_chunks):
            chunk, _ = sock.recvfrom(BUFFER_SIZE)
            chunks.append(chunk)

        # Gabungkan potongan dan decode
        buffer = b"".join(chunks)
        npdata = np.frombuffer(buffer, dtype=np.uint8)
        frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
        return frame
    except socket.timeout:
        print(f"[ERROR] timeout")

        return None
    except Exception as e:
        print(f"[ERROR] Receiving UDP stream: {e}")
        return None
    

def process_stream(model_path):
    model = YOLO(model_path, task="segment")
    fps_target = 30
    frame_delay = 1.0 / fps_target
    frame_id = 0
    print("1")
    prev_time = time.time()
    with open(RESULTS_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame ID", "FPS", "Detected Objects"])
        print("2")
        while True:
            start_time = time.time()
            print("3")
            frame = receive_udp_stream()
            if frame is None:
                continue

            results = model.predict(frame, conf=0.3, iou=0.2, imgsz=256, verbose=False)
            print("4")
            class_count = {}
            for out in results:
                for box in out.boxes:
                    label = out.names[int(box.cls.numpy()[0])]
                    class_count[label] = class_count.get(label, 0) + 1

            #summary = ", ".join([f"{v} {k}" for k, v in class_count.items()])
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()

            print(f"[INFO] Frame ID: - FPS: {fps:.2f}")
            #writer.writerow([fid, f"{fps:.2f}", summary])
            time.sleep(max(0, frame_delay - (time.time() - start_time)))

if __name__ == "__main__":
    YOLO_MODEL_PATH = "best_13-04-2025_full_integer_quant_edgetpu.tflite"
    process_stream(YOLO_MODEL_PATH)
