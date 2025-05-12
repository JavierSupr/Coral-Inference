import cv2
import time
import sys
import socket
import numpy as np
import struct
import csv
import json
from ultralytics import YOLO

# UDP configuration
RESULTS_DEST_IP = "192.168.137.1"  # IP perangkat penerima hasil inferensi
RESULTS_PORT = 5012                # Port tujuan untuk hasil inferensi
PORT_1 = 5015
RESULTS_CSV = "inference_results.csv"
BUFFER_SIZE = 65000
UDP_IP = "192.168.137.1"
video_port = 5019

results_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT_1))
sock.settimeout(1.0)

def receive_udp_stream():
    latest_data = None
    while True:
        try:
            data, addr = sock.recvfrom(65536)
            latest_data = data
            # Lanjutkan membaca sampai buffer kosong
            sock.setblocking(0)
            while True:
                try:
                    data, addr = sock.recvfrom(65536)
                    latest_data = data
                except BlockingIOError:
                    break
            sock.setblocking(1)  # Kembalikan blocking mode

            if latest_data is None:
                print("[WARNING] No data received")
                return None

            npdata = np.frombuffer(latest_data, dtype=np.uint8)
            frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
            return frame

        except socket.timeout:
            print(f"[ERROR] timeout")
            return None
        except Exception as e:
            print(f"[ERROR] Receiving UDP stream: {e}")
            return None


def process_stream(model_path, video_port):
    model = YOLO(model_path, task="segment")
    fps_target = 30
    frame_delay = 1.0 / fps_target
    fid = 0
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
            frame_copy = cv2.resize(frame,(256, 256))
            results = model.predict(frame_copy, conf=0.3, iou=0.2, imgsz=256, verbose=False)
            print("4")
            class_count = {}
            for out in results:
                for box in out.boxes:
                    label = out.names[int(box.cls.numpy()[0])]
                    class_count[label] = class_count.get(label, 0) + 1

            #summary = ", ".join([f"{v} {k}" for k, v in class_count.items()])
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            scale_x = 854 / 256
            scale_y = 480 / 256
            print(f"[INFO] Frame ID: {fid} - FPS: {fps:.2f}")
            #writer.writerow([fid, f"{fps:.2f}", summary])
            time.sleep(max(0, frame_delay - (time.time() - start_time)))
            objs_lst = []
            for out in results:
                masks = out.masks
                for i, box in enumerate(out.boxes):
                    cls = int(box.cls.numpy()[0])
                    conf = float(box.conf.numpy()[0])
                    label = out.names[cls]
                    bbox = box.xyxy.numpy()[0].tolist()
                    seg = masks.xy[i].tolist() if masks else []

                    bbox = [
                        bbox[0] * scale_x,  # x1
                        bbox[1] * scale_y,  # y1
                        bbox[2] * scale_x,  # x2
                        bbox[3] * scale_y   # y2
                    ]
            
                    # Scale segmentation coordinates back to original size
                    seg = []
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
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            # Split data into chunks
            chunks = [buffer[i:i + BUFFER_SIZE] for i in range(0, len(buffer), BUFFER_SIZE)]
            fid += 1
            try:
                results_sock.sendto(json.dumps(inference_data).encode(), (RESULTS_DEST_IP, RESULTS_PORT))
                
                video_sock.sendto(struct.pack("I", fid), (UDP_IP, video_port))  # Send frame ID
            
                video_sock.sendto(struct.pack("B", len(chunks)), (UDP_IP, video_port))
                for chunk in chunks:
                    video_sock.sendto(chunk, (UDP_IP, video_port))

            except Exception as e:
                print(f"[ERROR] Failed to send results: {e}")

if __name__ == "__main__":
    YOLO_MODEL_PATH = "best_13-04-2025_full_integer_quant_edgetpu.tflite"
    for arg in sys.argv[1:]:
        if arg.startswith("RESULTS_PORT="):
            try:
                RESULTS_PORT = int(arg.split("=")[1])
                print(f"[INFO] Using custom RESULTS_PORT: {RESULTS_PORT}")
            except ValueError:
                print("[ERROR] Invalid RESULTS_PORT number.")
                sys.exit(1)
        elif arg.startswith("PORT_1="):
            try:
                PORT_1 = int(arg.split("=")[1])
                print(f"[INFO] Using custom PORT_1: {PORT_1}")
            except ValueError:
                print("[ERROR] Invalid PORT_1 number.")
                sys.exit(1)
        elif arg.startswith("VIDEO_PORT="):
            try:
                video_port = int(arg.split("=")[1])
                print(f"[INFO] Using custom VIDEO_PORT: {video_port}")
            except ValueError:
                print("[ERROR] Invalid VIDEO_PORT number.")
                sys.exit(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT_1))
    sock.settimeout(1.0)

    process_stream(YOLO_MODEL_PATH, video_port)