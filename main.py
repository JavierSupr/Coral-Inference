import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def dequantize_output(output_data, output_details):
    """Dequantize the output from int8 to float"""
    zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
    scale = output_details[0]['quantization_parameters']['scales'][0]
    return (output_data - zero_point) * scale

def process_yolo_output(output_data, output_details, conf_threshold=0.25, iou_threshold=0.45):
    # Dequantize the output
    output = dequantize_output(output_data[0], output_details)
    
    # Reshape the output to [num_boxes, num_classes + 5]
    num_boxes = output.shape[1]
    num_classes = output.shape[2] - 5  # Subtract 5 for box coordinates and confidence
    detections = output[0].reshape(num_boxes, 5 + num_classes)
    
    # Filter by confidence threshold
    mask = detections[:, 4] >= conf_threshold
    filtered_detections = detections[mask]
    
    if len(filtered_detections) == 0:
        return []
    
    # Get boxes, scores, and classes
    boxes = filtered_detections[:, :4]  # [x, y, w, h]
    scores = filtered_detections[:, 4]
    class_ids = np.argmax(filtered_detections[:, 5:], axis=1)
    
    # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2  # y2
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        scores.tolist(),
        conf_threshold,
        iou_threshold
    )
    
    results = []
    for idx in indices:
        if isinstance(idx, list):  # Handle different OpenCV versions
            idx = idx[0]
        results.append({
            'bbox': boxes_xyxy[idx].tolist(),
            'score': float(scores[idx]),
            'class_id': int(class_ids[idx])
        })
    
    return results

def preprocess_frame(frame, input_shape):
    """Preprocess frame for model input"""
    image = cv2.resize(frame, (input_shape[1], input_shape[2]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Print debug information
    print(f"Frame min/max before normalization: {np.min(image)}, {np.max(image)}")
    
    # Normalize to [-1, 1] range for int8 quantization
    normalized_image = (image - 127.5) / 127.5
    input_data = np.expand_dims(normalized_image, axis=0).astype(np.int8)
    
    print(f"Input data min/max after normalization: {np.min(input_data)}, {np.max(input_data)}")
    return input_data

def main():
    # Initialize Edge TPU
    try:
        interpreter = tflite.Interpreter(
            model_path="best_full_integer_quant_edgetpu.tflite",
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        print("Edge TPU delegate loaded successfully.")
    except ValueError as e:
        print("Failed to load Edge TPU delegate:", e)
        return

    interpreter.allocate_tensors()

    print("Loaded delegates:", interpreter._delegates)
    print("Using Edge TPU:", "libedgetpu" in str(interpreter._delegates))

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    print(f"Input shape: {input_shape}")
    print("\nModel Details:")
    print("Input Details:", input_details)
    print("Output Details:", output_details)

    # Open video streams
    cap1 = cv2.VideoCapture('video.mp4')
    cap2 = cv2.VideoCapture('333 VID_20231011_170120.mp4')

    # Initialize timing variables
    prev_time1 = time.time()
    prev_time2 = time.time()

    try:
        while cap1.isOpened() and cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print("Failed to capture frames from one or both cameras.")
                break

            # Store original dimensions
            orig_h1, orig_w1 = frame1.shape[:2]
            orig_h2, orig_w2 = frame2.shape[:2]

            # Process first camera
            input_data1 = preprocess_frame(frame1, input_shape)
            interpreter.set_tensor(input_details[0]['index'], input_data1)
            
            start_time1 = time.time()
            interpreter.invoke()
            inference_time1 = time.time() - start_time1
            
            output_data1 = [interpreter.get_tensor(output_details[i]['index']) 
                           for i in range(len(output_details))]
            
            detections1 = process_yolo_output(output_data1, output_details)
            print(f"\nCamera 1 Detections: {len(detections1)}")
            for det in detections1:
                print(f"Class {det['class_id']}: {det['score']:.2f}")
                bbox = det['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Process second camera
            input_data2 = preprocess_frame(frame2, input_shape)
            interpreter.set_tensor(input_details[0]['index'], input_data2)
            
            start_time2 = time.time()
            interpreter.invoke()
            inference_time2 = time.time() - start_time2
            
            output_data2 = [interpreter.get_tensor(output_details[i]['index']) 
                           for i in range(len(output_details))]
            
            detections2 = process_yolo_output(output_data2, output_details)
            print(f"\nCamera 2 Detections: {len(detections2)}")
            for det in detections2:
                print(f"Class {det['class_id']}: {det['score']:.2f}")
                bbox = det['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate and display FPS
            current_time1 = time.time()
            current_time2 = time.time()
            
            if current_time1 - prev_time1 > 0:
                fps1 = 1 / (current_time1 - prev_time1)
                print(f"Camera 1 - FPS: {fps1:.2f}, Inference Time: {inference_time1 * 1000:.2f} ms")
            
            if current_time2 - prev_time2 > 0:
                fps2 = 1 / (current_time2 - prev_time2)
                print(f"Camera 2 - FPS: {fps2:.2f}, Inference Time: {inference_time2 * 1000:.2f} ms")


            prev_time1 = current_time1
            prev_time2 = current_time2


    except Exception as e:
        print(f"Error during processing: {e}")
        
    finally:
        # Clean up
        cap1.release()
        cap2.release()

if __name__ == "__main__":
    main()