import cv2
import socket
import numpy as np
import struct

# UDP Socket configuration
UDP_IP = "192.168.137.1"  # Replace with receiver's IP address
UDP_PORT = 5005         # Port to send data

# Initialize socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Open video capture
cap = cv2.VideoCapture("333 VID_20231011_170120_1.mp4")  # Use 0 for webcam, or provide a video file

# Set frame size
WIDTH = 640
HEIGHT = 480
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    # Split data into chunks (UDP max packet size is ~65507 bytes)
    max_packet_size = 65000
    chunks = [buffer[i:i + max_packet_size] for i in range(0, len(buffer), max_packet_size)]

    # Send number of chunks
    sock.sendto(struct.pack("B", len(chunks)), (UDP_IP, UDP_PORT))

    # Send each chunk
    for chunk in chunks:
        sock.sendto(chunk, (UDP_IP, UDP_PORT))

    # Display the sending video
    #cv2.imshow('Sender', frame)

    #f cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
cv2.destroyAllWindows()
sock.close()
