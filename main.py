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
    quantized = (normalized * 255).astype(np.uint8)  # Scale back to [0, 255] and convert to uint8
    return np.expand_dims(quantized, axis=0), width, height

# Postprocess output to extract bounding boxes and labels
def postprocess_output(interpreter, original_width, original_height, threshold=0.5):
    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[2]['index'])[0]
    
    results = []
    for i in range(len(scores)):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            results.append({
                'bounding_box': [
                    int(xmin * original_width),
                    int(ymin * original_height),
                    int(xmax * original_width),
                    int(ymax * original_height)
                ],
                'class_id': int(classes[i]),
                'score': scores[i]
            })
    return results

# Load label map
def load_labels(label_path):
    with open(label_path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def main():
    model_path = "best_full_integer_quant_edgetpu.tflite"
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
