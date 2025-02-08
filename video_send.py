import cv2
import socket
import numpy as np
import struct

# UDP Socket configuration
UDP_IP = "192.168.1.2"  # Replace with receiver's IP address
UDP_PORT = 5005         # Port to send data
BUFFER_SIZE = 65000     # Max UDP packet size

# Initialize socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Open video capture
cap = cv2.VideoCapture('/dev/video1')  # Use 0 for webcam, or provide a video file

# Set frame size
WIDTH = 640
HEIGHT = 480
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        # Split data into chunks (UDP max packet size is ~65507 bytes)
        chunks = [buffer[i:i + BUFFER_SIZE] for i in range(0, len(buffer), BUFFER_SIZE)]

        try:
            # Send number of chunks
            sock.sendto(struct.pack("B", len(chunks)), (UDP_IP, UDP_PORT))

            # Send each chunk
            for chunk in chunks:
                sock.sendto(chunk, (UDP_IP, UDP_PORT))

        except socket.error as e:
            print(f"[ERROR] Network issue: {e}")
            break  # Stop sending if network fails

except KeyboardInterrupt:
    print("[INFO] Stream interrupted by user.")

except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")

finally:
    # Cleanup resources
    cap.release()
    sock.close()
    print("[INFO] Stream closed.")
