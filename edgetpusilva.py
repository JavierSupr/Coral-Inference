from edge_tpu_silva import process_detection
from ultralytics import YOLO
model = YOLO()

# Run the object detection process
outs = process_detection(model_path='240_yolov8n_full_integer_quant_edgetpu.tflite', input_path='333-vid-20231011-170120_Tt2GmTrq.mp4', imgsz=240)

for _, _ in outs:
  pass