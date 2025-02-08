import cv2
import time
import threading
from ultralytics import YOLO


def process_segmentation(model_path, input_source, imgsz=256, threshold=0.4, fps_target=30):
    """Runs YOLO segmentation on a camera stream with dynamic frame skipping."""
    
    model = YOLO(model=model_path, task="segment")
    cap = cv2.VideoCapture(input_source)
    
    frame_skip = 1
    frame_count = 0
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of stream
        
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time
        desired_interval = 1.0 / fps_target
        
        if elapsed_time < desired_interval:
            frame_skip = max(1, frame_skip - 1)  # Reduce skipping if faster
        else:
            frame_skip = min(5, frame_skip + 1)  # Increase skipping if slower
        
        if frame_count % frame_skip == 0:
            prev_time = time.time()
            results = model.predict(frame, conf=threshold, imgsz=imgsz, verbose=False)
            
            objs_lst = []
            for out in results:
                masks = out.masks
                for index, box in enumerate(out.boxes):
                    seg = masks.xy[index]
                    obj_cls, conf, bb = (
                        box.cls.numpy()[0],
                        box.conf.numpy()[0],
                        box.xyxy.numpy()[0],
                    )
                    label = out.names[int(obj_cls)]
                    obj_data = {
                        "id": int(obj_cls),
                        "label": label,
                        "conf": float(conf),
                        "bbox": bb.tolist(),
                        "seg": [s.tolist() for s in seg],
                    }
                    objs_lst.append(obj_data)
            
            print(f"Camera {input_source}: FPS={1/elapsed_time:.2f}, Frame Skip: {frame_skip}, Detected Objects={len(objs_lst)}")
    
    cap.release()


def run_dual_camera_inference(model_path, cam1_source, cam2_source):
    """Runs YOLO segmentation on two cameras in parallel."""
    
    thread1 = threading.Thread(target=process_segmentation, args=(model_path, cam1_source))
    thread2 = threading.Thread(target=process_segmentation, args=(model_path, cam2_source))
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    print("Inference completed for both cameras.")


if __name__ == "__main__":
    YOLO_MODEL_PATH = "best_full_integer_quant_edgetpu.tflite"
    cam1_source = "333 VID_20231011_170120_1.mp4"
    cam2_source = "333-vid-20231011-170120_Tt2GmTrq.mp4"
    run_dual_camera_inference(YOLO_MODEL_PATH, cam1_source, cam2_source)
