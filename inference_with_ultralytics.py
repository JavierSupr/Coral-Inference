import time
from typing import Tuple, Union, List
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def preprocess_frame(frame, input_size=(640, 640)):
    """Preprocess frame for TFLite Edge TPU model."""
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

    return np.expand_dims(canvas, axis=0)  # Shape: (1, 640, 640, 3)

def load_labels(label_path="labels.txt"):
    """Load class labels from a file if provided."""
    try:
        with open(label_path, "r") as f:
            labels = {i: line.strip() for i, line in enumerate(f.readlines())}
        return labels
    except FileNotFoundError:
        return {}

def process_detection(
    model_path: str,
    input_path: str,
    imgsz: Union[int, Tuple[int, int]] = (640, 640),
    threshold: float = 0.4,
    verbose: bool = True,
    show: bool = False,
    classes: List[int] = None,
):
    """Run object detection on Coral Dev Board with Edge TPU TFLite model.

    Args:
        model_path (str): Path to the .tflite model (Edge TPU compiled).
        input_path (str): Path to input video/image or camera index (0,1,2).
        imgsz (Union[int, Tuple[int, int]]): Model's required input size.
        threshold (float, optional): Confidence threshold. Defaults to 0.4.
        verbose (bool, optional): Print detection results. Defaults to True.
        show (bool, optional): Display output frames. Defaults to False.
        classes (List[int], optional): Filter results by class ID. Defaults to None.

    Yields:
        frame, objs_lst, fps
    """

    # Load TFLite model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("1")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    labels = load_labels()  # Load labels if available

    # Check input type (image, video, or webcam)
    is_image = input_path.lower().endswith((".jpg", ".png", ".jpeg"))
    
    if is_image:
        cap = cv2.imread(input_path)
        video_mode = False
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video source {input_path}")
            return
        video_mode = True

    frame_count = 0
    start_time = time.time()

    while True:
        if video_mode:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = cap.copy()  # For single image processing

        processed_frame = preprocess_frame(frame, input_size=(input_shape[1], input_shape[2]))
        if processed_frame is None:
            continue

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], processed_frame)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        objs_lst = []
        for detection in output_data[0]:
            obj_cls, conf, x1, y1, x2, y2 = detection[:6]
            if conf < threshold:
                continue

            obj_cls = int(obj_cls)
            label = labels.get(obj_cls, f"Class {obj_cls}")

            obj_info = {
                "id": obj_cls,
                "label": label,
                "conf": float(conf),
                "bbox": [x1, y1, x2, y2],
            }
            objs_lst.append(obj_info)

            if verbose:
                print(f"Detected {label}: ID={obj_cls}, Score={conf:.2f}, BBox={x1,y1,x2,y2}")

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        if verbose:
            print(f"FPS: {fps:.2f}")

        # Draw bounding boxes
        for obj in objs_lst:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            label = f"{obj['label']} {obj['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        yield frame, objs_lst, fps

    if video_mode:
        cap.release()
    cv2.destroyAllWindows()
