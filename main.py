import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# Load YOLO TFLite model
def load_model(model_path):
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate("libedgetpu.so.1")]
    )
    interpreter.allocate_tensors()
    return interpreter

# Preprocess input frame for YOLO
def preprocess_frame(frame, input_size):
    height, width, _ = frame.shape
    resized = cv2.resize(frame, input_size)
    normalized = resized / 255.0  # Normalize pixel values to [0, 1]
    quantized = (normalized * 255).astype(np.int8)  # Scale back to [0, 255] and convert to uint8
    return np.expand_dims(quantized, axis=0), width, height

# Postprocess output to extract bounding boxes and labels
def postprocess_output(interpreter, original_width, original_height, threshold=0.5):
    output_details = interpreter.get_output_details()
    
    # Extract detections from PartitionedCall:0
    raw_output = interpreter.get_tensor(output_details[0]['index'])  # Shape [1, 116, 756]
    detections = raw_output[0]  # Remove batch dimension
    
    results = []
    for detection in detections:
        # Extract information from each detection
        bbox = detection[:4]  # First 4 values are bbox (normalized [ymin, xmin, ymax, xmax])
        confidence = detection[4]  # Confidence score
        class_scores = detection[5:]  # Class scores
        
        # Convert confidence and scores to float (dequantize)
        confidence = (confidence - 28) * 0.017405057325959206
        class_scores = (class_scores - 28) * 0.017405057325959206
        
        # Get the best class
        class_id = np.argmax(class_scores)
        class_score = class_scores[class_id]
        
        # Filter by threshold
        if confidence >= threshold:
            ymin, xmin, ymax, xmax = bbox
            results.append({
                'bounding_box': [
                    int(xmin * original_width),
                    int(ymin * original_height),
                    int(xmax * original_width),
                    int(ymax * original_height)
                ],
                'class_id': class_id,
                'score': confidence
            })
    return results


# Load label map
def load_labels(label_path):
    with open(label_path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def main():
    model_path = "240_yolov8n_full_integer_quant_edgetpu.tflite"
    label_path = "label.txt"
    video_path = "video.mp4"
    
    # Load model and labels
    interpreter = load_model(model_path)
    labels = load_labels(label_path)
    
    # Get input details
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape'][1:3]  # Height, Width
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        input_data, width, height = preprocess_frame(frame, input_shape)
        
        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Postprocess output
        results = postprocess_output(interpreter, width, height)
        
        # Print results
        print(f"Frame {frame_count}:")
        for result in results:
            class_id = result['class_id']
            score = result['score']
            bbox = result['bounding_box']
            print(f" - Class: {labels[class_id]}, Score: {score:.2f}, BBox: {bbox}")
        
        frame_count += 1
    
    # Cleanup
    cap.release()

if __name__ == "__main__":
    main()