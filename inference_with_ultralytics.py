import cv2
import torch
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, render_template
from datetime import datetime

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO(model="best_full_integer_quant_edgetpu.tflite", task="segment")  # Adjust model path if necessary
threshold = 0.3  # Confidence threshold
imgsz = 256  # Image size

video_path = "333-vid-20231011-170120_Tt2GmTrq.mp4"  # Change this to your video file
cap = cv2.VideoCapture(video_path)

def generate_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(frame, conf=threshold, imgsz=imgsz, verbose=False)
        inference_timestamp = datetime.now().strftime("%H:%M:%S.%f")
        
        for out in results:
            for box in out.boxes:
                obj_cls, conf, bb = (
                    int(box.cls.numpy()[0]),
                    float(box.conf.numpy()[0]),
                    box.xyxy.numpy()[0],
                )
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(
                #    frame,
                #    f"{model.names[obj_cls]}: {conf:.2f}",
                #    (x1, y1 - 10),
                #    cv2.FONT_HERSHEY_SIMPLEX,
                #    0.5,
                #    (0, 255, 0),
                #    2,
                #)
        
        # Encode frame
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "<h1>YOLOv8 Video Streaming</h1><img src='/video_feed' width='640px'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
