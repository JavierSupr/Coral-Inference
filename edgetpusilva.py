import time
import cv2
import threading
from flask import Flask, Response
from ultralytics import YOLO
from typing import Tuple, Union, List

app = Flask(__name__)
model_path = "best_full_integer_quant_edgetpu.tflite"
input_path = "333-vid-20231011-170120_Tt2GmTrq.mp4"
imgsz = 256
threshold = 0.4
verbose = True
show = False
classes = None

def process_segmentation(
    model_path: str,
    input_path: str,
    imgsz: Union[int, Tuple[int, int]],
    threshold: float = 0.4,
    verbose: bool = True,
    show: bool = False,
    classes: List[int] = None,
):
    """Run object segmentation with edge-tpu-silva"""

    model = YOLO(model=model_path, task="segment")
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
        frame = out.plot()
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        objs_lst = []
        for index, box in enumerate(out.boxes):
            seg = out.masks.xy[index]
            obj_cls, conf, bb = (
                box.cls.numpy()[0],
                box.conf.numpy()[0],
                box.xyxy.numpy()[0],
            )
            label = out.names[int(obj_cls)]
            objs_lst.append({"id": obj_cls, "label": label, "conf": conf, "bbox": bb, "seg": seg})

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        if verbose:
            print(f"\nDetected Objects: {objs_lst}")
            print(f"FPS: {fps:.2f}")

@app.route('/')
def index():
    return """<h1>YOLOv8 Video Streaming</h1><img src='/video_feed' width='720px'>"""

@app.route('/video_feed')
def video_feed():
    return Response(process_segmentation(model_path, input_path, imgsz, threshold, verbose, show, classes),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_segmentation():
    for objects, fps in process_segmentation(model_path, input_path, imgsz, threshold, verbose, show, classes):
        print("Detected Objects:", objects)
        print("FPS:", fps)

if __name__ == '__main__':
    threading.Thread(target=run_segmentation, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)