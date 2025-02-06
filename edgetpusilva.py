import time
import socket
import json
from typing import Tuple, Union, List

import cv2
from ultralytics import YOLO

# UDP Configuration
UDP_IP = "192.168.137.1"  # Replace with your PC's IP address
UDP_PORT = 5000  # Port to send data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create UDP socket

def process_segmentation(
    model_path: str,
    input_path: str,
    imgsz: Union[int, Tuple[int, int]],
    threshold: float = 0.4,
    verbose: bool = True,
    show: bool = False,
    classes: List[int] = None,
):
    """Run object segmentation with edge-tpu and send results via UDP."""

    # Load a model
    model = YOLO(model=model_path, task="segment")

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
        
        masks = out.masks

        if verbose:
            print("\n\n-------RESULTS--------")
        objs_lst = []
        for index, box in enumerate(out.boxes):
            seg = masks.xy[index]
            obj_cls, conf, bb = (
                box.cls.numpy()[0],
                box.conf.numpy()[0],
                box.xyxy.numpy()[0],
            )
            label = out.names[int(obj_cls)]
            ol = {"id": int(obj_cls), "label": label, "conf": float(conf), "bbox": bb.tolist(), "seg": [s.tolist() for s in seg]}
            objs_lst.append(ol)

            if verbose:
                print(label)
                print("  id:    ", obj_cls)
                print("  score: ", conf)
                print("  seg:  ", type(seg))

        frame_count += 1

        if verbose:
            print("\n----INFERENCE TIME----")
            print("FPS: {:.2f}".format(fps))

        # Convert objects to JSON and send via UDP
        message = json.dumps({"objects": objs_lst, "fps": fps})
        sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

        # Break the loop if 'esc' key is pressed for video or camera
        if cv2.waitKey(1) == 27:
            break

        yield objs_lst, fps


# Define paths and parameters
model_path = "best_full_integer_quant_edgetpu.tflite"
input_path = "333 VID_20231011_170120_1.mp4"
imgsz = 256
threshold = 0.4
verbose = True
show = False
classes = None

# Call the function and process the output
for objects, fps in process_segmentation(
    model_path, input_path, imgsz, threshold, verbose, show, classes
):
    print("\nDetected Objects:", objects)
    print("FPS:", fps)
