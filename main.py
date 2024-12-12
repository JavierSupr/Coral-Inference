import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Video input (bisa dari file atau kamera)
video_path = "333 VID_20231011_170120.mp4"  # Path ke video, atau gunakan 0 untuk kamera
cap = cv2.VideoCapture(video_path)

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)  # Sesuaikan tipe jika perlu
    input_data = input_data / 255.0  # Normalisasi

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()

    # Get results
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process detection results
    for detection in output_data:
        confidence = detection[1]  # Confidence score
        if confidence > 0.5:  # Threshold deteksi
            bbox = detection[2:6]  # Bounding box: xmin, ymin, xmax, ymax
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Gambarkan bounding box dan label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Kotak hijau
            label = f"Confidence: {confidence:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan frame dengan deteksi
    cv2.imshow("Deteksi YOLO", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
