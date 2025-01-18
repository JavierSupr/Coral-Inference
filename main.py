import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    default_model_dir = '../all_models'
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

        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]

        print(f"Frame {frame_index}:")
        for obj in objs:
            bbox = obj.bbox
            label = labels.get(obj.id, obj.id)
            print(f"- Label: {label}, Score: {obj.score:.2f}, Bounding Box: ({bbox.xmin}, {bbox.ymin}, {bbox.xmax}, {bbox.ymax})")

        frame_index += 1

    cap.release()

if __name__ == '__main__':
    main()
