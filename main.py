from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model for segmentation
model = YOLO('best.pt')  # Replace with the appropriate model file

# Load a video to segment
video_path = '111 VID_20231011_170829.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if the video is loaded successfully
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to format supported by Coral TPU (if needed)
    input_frame = cv2.resize(frame, (300, 300))  # Resize to match TPU requirements
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(input_frame, axis=0).astype(np.uint8)  # Convert to tensor

    # Perform segmentation using Coral TPU
    results = model.predict(input_tensor, task='segment')

    # Visualize the segmentation result
    segmented_frame = results[0].plot()  # Plot segmentation results

    # Display the segmented frame
    cv2.imshow('YOLOv8 Segmentation on Coral TPU', segmented_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
