import cv2
import time
import sys
import socket
import numpy as np
import struct
import csv
import json
import threading
from ultralytics import YOLO

BUFFER_SIZE = 65000

def receive_udp_stream(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(1.0)
    latest_data = None
    while True:
        try:
            data, addr = sock.recvfrom(65536)
            latest_data = data
            sock.setblocking(0)
            while True:
                try:
                    data, addr = sock.recvfrom(65536)
                    latest_data = data
                except BlockingIOError:
                    break
            sock.setblocking(1)

            if latest_data is None:
                print("[WARNING] No data received")
                return None

            npdata = np.frombuffer(latest_data, dtype=np.uint8)
            frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
            return frame

        except socket.timeout:
            return None
        except Exception as e:
            print(f"[ERROR] Receiving UDP stream on port {port}: {e}")
            return None

def process_stream(model_path, udp_input_port, results_ip, results_port, video_ip, video_port, csv_filename):
    model = YOLO(model_path, task="segment")

    fps_target = 30
    frame_delay = 1.0 / fps_target
    fid = 0
    prev_time = time.time()

    results_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame ID", "FPS", "Detected Objects"])
        while True:
            start_time = time.time()
            frame = receive_udp_stream(udp_input_port)
            if frame is None:
                continue

            frame_copy = cv2.resize(frame, (256, 256))
            results = model.predict(frame_copy, conf=0.3, iou=0.2, imgsz=256, verbose=False)

            class_count = {}
            for out in results:
                for box in out.boxes:
                    label = out.names[int(box.cls.numpy()[0])]
                    class_count[label] = class_count.get(label, 0) + 1

            summary = ", ".join([f"{v} {k}" for k, v in class_count.items()])
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()

            writer.writerow([fid, f"{fps:.2f}", summary])

            objs_lst = []
            scale_x = 854 / 256
            scale_y = 480 / 256
            for out in results:
                masks = out.masks
                for i, box in enumerate(out.boxes):
                    cls = int(box.cls.numpy()[0])
                    conf = float(box.conf.numpy()[0])
                    label = out.names[cls]
                    bbox = box.xyxy.numpy()[0].tolist()
                    seg = masks.xy[i].tolist() if masks else []

                    bbox = [
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y
                    ]
                    if masks:
                        seg = [[x * scale_x, y * scale_y] for x, y in masks.xy[i]]

                    obj_data = {
                        "frame_id": fid,
                        "label": label,
                        "class_id": cls,
                        "conf": conf,
                        "bbox": bbox,
                        "seg": seg
                    }
                    objs_lst.append(obj_data)

            inference_data = {
                "frame_id": fid,
                "timestamp": time.time(),
                "objects": objs_lst
            }

            try:
                results_sock.sendto(json.dumps(inference_data).encode(), (results_ip, results_port))
            except Exception as e:
                print(f"[ERROR] Failed to send results: {e}")

            try:
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                chunks = [buffer[i:i + BUFFER_SIZE] for i in range(0, len(buffer), BUFFER_SIZE)]
                video_sock.sendto(struct.pack("I", fid), (video_ip, video_port))
                video_sock.sendto(struct.pack("B", len(chunks)), (video_ip, video_port))
                for chunk in chunks:
                    video_sock.sendto(chunk, (video_ip, video_port))
            except Exception as e:
                print(f"[ERROR] Failed to send video: {e}")

            fid += 1
            time.sleep(max(0, frame_delay - (time.time() - start_time)))


if __name__ == "__main__":
    YOLO_MODEL_PATH = "best_13-04-2025_full_integer_quant_edgetpu.tflite"

    # Konfigurasi untuk kamera 1
    cam1_config = {
        "udp_input_port": 5016,
        "results_ip": "192.168.137.1",
        "results_port": 5013,
        "video_ip": "192.168.137.1",
        "video_port": 5020,
        "csv_filename": "inference_results_cam1.csv"
    }

    # Konfigurasi untuk kamera 2
    cam2_config = {
        "udp_input_port": 5015,
        "results_ip": "192.168.137.1",
        "results_port": 5022,
        "video_ip": "192.168.137.1",
        "video_port": 5029,
        "csv_filename": "inference_results_cam2.csv"
    }

    # Jalankan thread untuk masing-masing kamera
    thread1 = threading.Thread(target=process_stream, kwargs={"model_path": YOLO_MODEL_PATH, **cam1_config})
    thread2 = threading.Thread(target=process_stream, kwargs={"model_path": YOLO_MODEL_PATH, **cam2_config})

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
