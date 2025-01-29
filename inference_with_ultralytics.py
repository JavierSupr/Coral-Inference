import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects

def preprocess_frame(frame, input_size=(256, 256)):  # Changed input size to 256x256
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

    input_mean = 128
    input_std = 128
    int8_input = ((canvas.astype(np.float32) - input_mean) / input_std).astype(np.int8)

    return np.expand_dims(int8_input, axis=0)

def process_yolo_output(output_data, conf_threshold=0.25):
    # Assuming YOLO output format: [batch, num_boxes, num_classes + 5]
    # Where 5 represents [x, y, w, h, confidence]
    boxes = []
    class_ids = []
    confidences = []
    
    # Reshape output if necessary
    predictions = output_data[0]  # Take first batch
    
    for i, detection in enumerate(predictions):
        print(f"{i}detection{detection}")
        confidence = detection[4]  # Object confidence
        print(f"conf{confidence}")
        if confidence > conf_threshold:
            class_scores = detection[5:]  # Class scores
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            # Only proceed if class confidence is good
            if class_confidence > conf_threshold:
                # Extract bounding box coordinates
                x, y, w, h = detection[0:4]
                
                boxes.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence * class_confidence))
    
    return boxes, class_ids, confidences

def main():
    model_path = "240_yolov8n-seg_full_integer_quant_edgetpu.tflite"
    video_path = "333 VID_20231011_170120.mp4"

    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print input shape information for verification
    input_shape = input_details[0]['shape']
    print(f"Model expects input shape: {input_shape}")

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

        # Get detection output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Process YOLO output
        boxes, class_ids, confidences = process_yolo_output(output_data)
        print(output_data[0])
        print()
        # Print results for this frame
        #print(f"\nFrame {frame_count} Detections:")
        #for i in range(len(boxes)):
        #    print(f"Detection {i + 1}:")
        #    print(f"  Class ID: {class_ids[i]}")
        #    print(f"  Confidence: {confidences[i]:.2f}")
        #    print(f"  Bounding Box: x={boxes[i][0]:.2f}, y={boxes[i][1]:.2f}, w={boxes[i][2]:.2f}, h={boxes[i][3]:.2f}")

        frame_count += 1

    cap.release()
    print(f"\nProcessed {frame_count} frames")

if __name__ == "__main__":
    main()