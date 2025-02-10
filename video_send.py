import cv2
import socket
import numpy as np
import struct
import threading
import time

# UDP Socket configuration
UDP_IP = "192.168.137.1"  # Replace with receiver's IP address
PORT_1 = 5005           # Port for first video stream
PORT_2 = 5006           # Port for second video stream
BUFFER_SIZE = 65000     # Max UDP packet size

# Initialize sockets for two video streams
sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Open two video sources (change these if using files instead)
cap1 = cv2.VideoCapture('333 VID_20231011_170120_1.mp4')  # First camera (or replace with "video1.mp4")
cap2 = cv2.VideoCapture('333 VID_20231011_170120_1.mp4')  # Second camera (or replace with "video2.mp4")

# Set frame size for both streams
WIDTH = 640
HEIGHT = 480
cap1.set(3, WIDTH)
cap1.set(4, HEIGHT)
cap2.set(3, WIDTH)
cap2.set(4, HEIGHT)

# Get frame rate of video
fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if unavailable
fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30

def stream_video(cap, sock, port, stream_name, fps):
    """Function to capture and send a video stream over UDP."""
    frame_delay = 1.0 / fps  # Calculate delay per frame

    try:
        while cap.isOpened():
            start_time = time.time()  # Track time at start of frame capture

            ret, frame = cap.read()
            if not ret:
                print(f"[ERROR] Failed to read frame from {stream_name}.")
                break

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

    except KeyboardInterrupt:
        print(f"[INFO] {stream_name} stream interrupted by user.")

    except Exception as e:
        print(f"[ERROR] Unexpected error in {stream_name}: {e}")

    finally:
        # Cleanup resources
        cap.release()
        sock.close()
        print(f"[INFO] {stream_name} stream closed.")


# Start both streams in separate threads
thread1 = threading.Thread(target=stream_video, args=(cap1, sock1, PORT_1, "Camera 1", fps1))
thread2 = threading.Thread(target=stream_video, args=(cap2, sock2, PORT_2, "Camera 2", fps2))

thread1.start()
thread2.start()

thread1.join()
thread2.join()
