import cv2
import torch
import numpy as np
import time
from flask import Flask, Response, render_template
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best_full_integer_quant_edgetpu.tflite", task= 'segment')  # Replace with your trained model

# Flask app
app = Flask(__name__)

# Video source (use 0 for webcam or replace with video file path)
VIDEO_SOURCE = "333-vid-20231011-170120_Tt2GmTrq.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Threshold and image size
CONF_THRESHOLD = 0.5
IMGSZ = 256

CLASS_COLORS = {
    0: (0, 255, 0),    # Green
    1: (255, 0, 0),    # Blue
    2: (0, 0, 255)     # Red
}

def generate_frames():
    prev_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Measure time to calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Run YOLOv8 inference
        results = model.predict(frame, conf=CONF_THRESHOLD, iou=0.3, imgsz=IMGSZ, verbose=False, stream=True)

        for out in results:
            masks = out.masks
            for index, box in enumerate(out.boxes):
                seg = masks.xy[index] if masks else None
                obj_cls, conf, bb = (
                    int(box.cls.numpy()[0]),
                    float(box.conf.numpy()[0]),
                    box.xyxy.numpy()[0],
                )

                # Get class-specific color (default to white if class not found)
                box_color = CLASS_COLORS.get(obj_cls, (255, 255, 255))

                # Draw bounding box
                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Draw label
                label = f"ID: {obj_cls} ({conf:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2,
                )

                # Draw segmentation mask (if available)
                #if seg is not None:
                #    seg = np.array(seg, np.int32)
                #   cv2.polylines(frame, [seg], isClosed=True, color=box_color, thickness=2)

        # Draw FPS counter
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")



@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    return Response(generate_frames(args.video), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
