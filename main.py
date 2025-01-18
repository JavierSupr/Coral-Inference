import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def yolo_post_process(output_tensor, labels, threshold):
    """
    Process YOLO output tensor to extract bounding boxes, class scores, and labels.
    """
    # Remove batch dimension
    predictions = np.squeeze(output_tensor)

    # Parse each prediction
    results = []
    for pred in predictions.T:  # Transpose to iterate over each prediction
        class_scores = pred[4:]  # Assuming the first 4 values are bounding box
        max_class_score = np.max(class_scores)
        class_id = np.argmax(class_scores)

        if max_class_score > threshold:
            bbox = pred[:4]  # First 4 are bounding box coordinates
            results.append({
                "bbox": bbox,
                "class_id": class_id,
                "score": max_class_score,
                "label": labels.get(class_id, "Unknown")
            })

    return results

def main():
    default_model_dir = ''
    default_model = '240_yolov8n_full_integer_quant_edgetpu.tflite'
    default_labels = 'label.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--input', type=str, 
                        default='video.mp4')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    # Open video source
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))  # Use camera index
    else:
        cap = cv2.VideoCapture(args.input)  # Use video file

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to model input size
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

        # Run inference
        run_inference(interpreter, cv2_im_rgb.tobytes())

        # Extract and process output tensor
        output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
        results = yolo_post_process(output_tensor, labels, args.threshold)

        # Print results for the current frame
        print(f"Frame {frame_index}:")
        for result in results:
            bbox = result["bbox"]
            print(f"- Label: {result['label']}, Score: {result['score']:.2f}, Bounding Box: {bbox}")

        frame_index += 1

    cap.release()

if __name__ == '__main__':
    main()
