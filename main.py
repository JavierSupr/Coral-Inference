import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class YOLODetector:
    def __init__(self, model_path, labels_path, confidence_threshold=0.5):
        # Load TFLite model
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]  # Coral TPU acceleration
        )
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        self.confidence_threshold = confidence_threshold

    def preprocess_frame(self, frame, input_shape=(416, 416)):
        # Resize and normalize frame
        resized = cv2.resize(frame, input_shape)
        normalized = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(normalized, axis=0)
        return input_data

    def detect_objects(self, frame):
        # Preprocess input
        input_data = self.preprocess_frame(frame)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensors
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        scores = self.interpreter.get_tensor(self.output_details[1]['index'])
        classes = self.interpreter.get_tensor(self.output_details[2]['index'])
        
        # Process detections
        valid_detections = []
        for i in range(len(scores[0])):
            if scores[0][i] > self.confidence_threshold:
                class_id = int(classes[0][i])
                bbox = boxes[0][i]
                
                # Convert normalized coordinates to pixel coordinates
                height, width = frame.shape[:2]
                x1 = int(bbox[1] * width)
                y1 = int(bbox[0] * height)
                x2 = int(bbox[3] * width)
                y2 = int(bbox[2] * height)
                
                valid_detections.append({
                    'label': self.labels[class_id],
                    'confidence': float(scores[0][i]),
                    'bbox': (x1, y1, x2, y2)
                })
        
        return valid_detections

    def draw_detections(self, frame, detections):
        for detection in detections:
            label = f"{detection['label']}: {detection['confidence']:.2f}"
            x1, y1, x2, y2 = detection['bbox']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put label text
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame

def process_video(video_path, model_path, labels_path):
    # Initialize detector
    detector = YOLODetector(
        model_path=model_path, 
        labels_path=labels_path
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detector.detect_objects(frame)
        
        # Draw detections
        result_frame = detector.draw_detections(frame, detections)
        
        # Display or save frame
        cv2.imshow('YOLO Detections', result_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    process_video(
        video_path='333 VID_20231011_170120.mp4',
        model_path='best_float16.tflite',
        labels_path='label.txt'
    )