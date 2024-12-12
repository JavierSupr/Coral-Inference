import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
from pycoral.adapters import common
from pycoral.adapters import detect

class YOLOv8Segmentation:
    def __init__(self, model_path, labels_path=None):
        # Initialize the EdgeTPU interpreter
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        
        # Input and output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load labels if provided
        self.labels = None
        if labels_path:
            self.labels = read_label_file(labels_path)
    
    def preprocess_image(self, image):
        # Resize and normalize image for model input
        input_shape = self.input_details[0]['shape'][1:3]
        preprocessed = cv2.resize(image, input_shape)
        preprocessed = preprocessed.astype(np.float32) / 255.0
        preprocessed = preprocessed[np.newaxis, :]
        return preprocessed
    
    def process_video(self, video_path):
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            input_data = self.preprocess_image(frame)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get segmentation results
            segmentation_results = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process and visualize results
            for result in segmentation_results:
                # Draw segmentation mask
                mask = (result[..., 4:] > 0.5).astype(np.uint8)
                
                for i in range(mask.shape[2]):
                    if mask[0, :, :, i].sum() > 0:
                        # Generate a random color for this segment
                        color = np.random.randint(0, 255, size=3)
                        segment = mask[0, :, :, i]
                        
                        # Blend the colored segment with the original frame
                        colored_segment = np.zeros_like(frame)
                        colored_segment[segment == 1] = color
                        
                        # Apply some transparency
                        alpha = 0.5
                        cv2.addWeighted(colored_segment, alpha, frame, 1 - alpha, 0, frame)
            
            # Display the frame
            cv2.imshow('YOLOv8 Segmentation', frame)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

# Usage example
def main():
    model_path = 'best_float16.tflite'
    video_path = '333 VID_20231011_170120.mp4'
    
    segmenter = YOLOv8Segmentation(model_path)
    segmenter.process_video(video_path)

if __name__ == '__main__':
    main()