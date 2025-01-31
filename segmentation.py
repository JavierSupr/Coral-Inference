import argparse
import cv2
import numpy as np
from PIL import Image
from flask import Flask, Response
from pycoral.adapters import common, segment
from pycoral.utils.edgetpu import make_interpreter

def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3
    return colormap

def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]

def process_frame(frame, interpreter, width, height):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_img = img.resize((width, height), Image.LANCZOS)
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    result = segment.get_output(interpreter)
    if len(result.shape) == 3:
        result = np.argmax(result, axis=-1)
    mask = label_to_color_image(result).astype(np.uint8)
    mask_img = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 0.6, mask_img, 0.4, 0)

def generate_frames(interpreter, width, height, video_source):
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        segmented_frame = process_frame(frame, interpreter, width, height)
        ret, buffer = cv2.imencode('.jpg', segmented_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path of the segmentation model.', default= 'deeplabv3_mnv2_pascal_quant.tflite')
    parser.add_argument('--input', required=True, help='File path of the input video.')
    args = parser.parse_args()

    interpreter = make_interpreter(args.model, device=':0')
    interpreter.allocate_tensors()
    width, height = common.input_size(interpreter)

    app = Flask(__name__)

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(interpreter, width, height, args.input), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

if __name__ == '__main__':
    main()
