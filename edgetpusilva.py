import time
from typing import Tuple, Union, List

import cv2
from ultralytics import YOLO


def process_detection(
    model_path: str,
    input_path: str,
    imgsz: Union[int, Tuple[int, int]],
    threshold: float = 0.4,
    verbose: bool = True,
    show: bool = False,
    classes: List[int] = None,
):
    # Load a model
    model = YOLO(model=model_path, task="detect")

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
    prev_time = time.time()
    
    for out in outs:
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time
        
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        
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

        if verbose:
            print("----INFERENCE TIME----")
            print("FPS: {:.2f}".format(fps))

        yield objs_lst, fps

        # Break the loop if 'esc' key is pressed for video or camera
        if cv2.waitKey(1) == 27:
            break

# Define paths and parameters
model_path = "192_yolov8n_full_integer_quant_edgetpu.tflite"
input_path = "333-vid-20231011-170120_Tt2GmTrq.mp4"
imgsz = 256
threshold = 0.4
verbose = True
show = False
classes = None

# Call the function and process the output
for objects, fps in process_detection(
    model_path, input_path, imgsz, threshold, verbose, show, classes
):
    print("\nDetected Objects:", objects)
    print("FPS:", fps)
