from pycoral.utils.edgetpu import make_interpreter

# Load the TFLite model
model_path = "yolov8n_full_integer_quant_edgetpu.tflite"

# Load model interpreter
interpreter =make_interpreter(model_path)
interpreter.allocate_tensors()

# Get model details
try:
    metadata = interpreter.get_signature_list()
    print("Model Metadata:", metadata)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output_shape = output_details[0]['shape']
    num_classes = output_shape[-1] - 5  # Subtract 5 (box coordinates)
    print("Number of Classes:", num_classes)
    print("Input Details:", input_details)
    print("Output Details:", output_details)
except AttributeError:
    print("Metadata not found in this TFLite model.")
