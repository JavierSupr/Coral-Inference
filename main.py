import argparse
import cv2
import os
import time
import numpy as np
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


def draw_results(frame, results, labels):
    """
    Draw detection results on the frame.
    """
    for result in results:
        bbox = result["bbox"]
        label = result["label"]
        score = result["score"]

        # Convert normalized bbox to pixel values
        height, width, _ = frame.shape
        xmin, ymin, xmax, ymax = (
            int(bbox[0] * width),
            int(bbox[1] * height),
            int(bbox[2] * width),
            int(bbox[3] * height),
        )

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Draw label and score
        label_text = f"{label}: {score:.2f}"
        cv2.putText(
            frame,
            label_text,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return frame


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
    total_time = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to model input size
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

        # Measure inference time
        start_inference = time.time()
        run_inference(interpreter, cv2_im_rgb.tobytes())
        inference_time = time.time() - start_inference

        # Extract and process output tensor
        output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
        results = yolo_post_process(output_tensor, labels, args.threshold)

        # Draw results on the frame
        frame_with_results = draw_results(frame, results, labels)

        # Display the frame
        cv2.imshow('YOLO Detection', frame_with_results)

        # Print results for the current frame
        print(f"Frame {frame_index}:")
        for result in results:
            bbox = result["bbox"]
            print(f"- Label: {result['label']}, Score: {result['score']:.2f}, Bounding Box: {bbox}")
        print(f"Inference Time: {inference_time:.2f} seconds")

        frame_index += 1
        total_time = time.time() - start_time

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    fps = frame_index / total_time if total_time > 0 else 0
    print(f"Processed {frame_index} frames in {total_time:.2f} seconds. FPS: {fps:.2f}")


if __name__ == '__main__':
    main()
