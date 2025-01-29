import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def preprocess_frame(frame, input_size=(640, 640)):
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
    return np.expand_dims(canvas, axis=0)  # Shape: (1, 640, 640, 3)

def main():
    model_path = "yolov8n_edgetpu.tflite"
    video_path = "video.mp4"

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

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

        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Frame {frame_count}: Detection Output ->", output_data)

        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    main()
