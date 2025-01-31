import cv2
import numpy as np
from flask import Flask, Response
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size

def preprocess_frame(frame, input_size=(513, 513)):
    if frame is None or frame.size == 0:
        return None
    
    height, width = frame.shape[:2]
    scale = min(input_size[0] / width, input_size[1] / height)
    new_width = min(int(width * scale), input_size[0])
    new_height = min(int(height * scale), input_size[1])
    
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    
    y_offset = max(0, (input_size[1] - new_height) // 2)
    x_offset = max(0, (input_size[0] - new_width) // 2)
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    # Convert to UINT8 (Edge TPU model requires UINT8)
    return np.expand_dims(canvas, axis=0)  # No need for normalization, keep raw uint8

def generate_frames():
    model_path = "mobilenetv2_deeplabv3_edgetpu.tflite"  # Change to correct model path
    video_path = "333 VID_20231011_170120.mp4"

    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        if processed_frame is None:
            continue

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], processed_frame)
        interpreter.invoke()

        # Get segmentation output (usually a mask)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        segmentation_mask = output_data[0]  # Extract mask from batch
        
        # Overlay segmentation mask on frame
        mask_resized = cv2.resize(segmentation_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        frame[mask_resized > 0] = [0, 255, 0]  # Apply green mask
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
