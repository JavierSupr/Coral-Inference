import argparse
import cv2
import numpy as np
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference

def save_output_tensor(output_tensor, filename="output_tensor.txt"):
    reshaped_tensor = output_tensor.reshape(116, 1344)  # Reshape to 116 rows, 1344 columns
    np.savetxt(filename, reshaped_tensor, delimiter=',', fmt='%.6f')
    print(f"Output tensor saved to {filename}")

def process_image(args):
    print(f'Loading {args.model} with {args.labels}')
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    
    image = cv2.imread(args.input)
    if image is None:
        print("Error: Unable to open image source.")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inference_size = tuple(interpreter.get_input_details()[0]['shape'][1:3])
    image_resized = cv2.resize(image_rgb, inference_size)
    
    run_inference(interpreter, image_resized.tobytes())
    output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
    
    print(f"Output tensor shape: {output_tensor.shape}")
    save_output_tensor(output_tensor)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n_full_integer_quant_edgetpu.tflite')
    parser.add_argument('--labels', default='label.txt')
    parser.add_argument('--input', type=str, default='test.jpg')
    args = parser.parse_args()
    process_image(args)

if __name__ == '__main__':
    main()