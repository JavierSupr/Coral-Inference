import tensorflow as tf

# Load the TFLite model
model_path = "yolov8n_full_integer_quant_edgetpu.tflite"

# Load model interpreter
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model details
try:
    metadata = interpreter.get_signature_list()
    print("Model Metadata:", metadata)
except AttributeError:
    print("Metadata not found in this TFLite model.")
