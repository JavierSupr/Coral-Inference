import time
from typing import Tuple, Union, List

import cv2
from ultralytics import YOLO


def process_detection(
    model_path: str,
    input_path: str,
    imgsz: Union[int, Tuple[int, int]],
    threshold: int = 0.4,
    verbose: bool = True,
    show: bool = False,
    classes: List[int] = None,
):

    # Load a model
    model = YOLO(
        model=model_path, task="detect"
    )  # Load a official model or custom model

    # Run Prediction
    outs = model.predict(
        source=input_path,
        conf=threshold,
        imgsz=imgsz,
        verbose=False,
        stream=True,
        show=show,
        classes=classes,
    )

    frame_count = 0
    start_time = time.time()
    for out in outs:
        if verbose:
            print("\n\n-------RESULTS--------")
        objs_lst = []
        for box in out.boxes:
            obj_cls, conf, bb = (
                box.cls.numpy()[0],
                box.conf.numpy()[0],
                box.xyxy.numpy()[0],
            )
            label = out.names[int(obj_cls)]
            ol = {
                "id": obj_cls,
                "label": label,
                "conf": conf,
                "bbox": bb,
            }
            objs_lst.append(ol)

            if verbose:
                print(label)
                print("  id:    ", obj_cls)
                print("  score: ", conf)
                print("  bbox:  ", bb)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        if verbose:
            print("----INFERENCE TIME----")
            print("FPS: {:.2f}".format(fps))

        yield objs_lst, fps

        # Break the loop if 'esc' key is pressed for video or camera
        if cv2.waitKey(1) == 27:
            break

# Define paths and parameters
model_path = "240_yolov8n_full_integer_quant_edgetpu.tflite"  # Path to the YOLO model (change to your .tflite model if needed)
input_path = "333-vid-20231011-170120_Tt2GmTrq.mp4"  # Path to an image or video file; use '0' for webcam
imgsz = 640  # Image size used during inference
threshold = 0.4  # Confidence threshold
verbose = True  # Print detected objects
show = False  # Display detection results
classes = None  # Detect all classes, or specify class IDs (e.g., [0, 1, 2])

# Call the function and process the output
for objects, fps in process_detection(
    model_path, input_path, imgsz, threshold, verbose, show, classes
):
    print("\nDetected Objects:", objects)
    print("FPS:", fps)
