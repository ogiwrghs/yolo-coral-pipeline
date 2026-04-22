import cv2
import numpy as np
from flask import Flask, Response
import os
import time
import tflite_runtime.interpreter as tflite
import logging

app = Flask(__name__)

MODELPATH = '/path/to/your/compiled_model_edgetpu.tflite'
IMGSIZE = 416    

cv2.setNumThreads(1)

try:
    edgetpudelegate = tflite.load_delegate('libedgetpu.so.1.0')
except ValueError:
    edgetpudelegate = tflite.load_delegate('libedgetpu.so.1')

interpreter = tflite.Interpreter(model_path=MODELPATH, experimental_delegates=[edgetpudelegate], num_threads=4)
interpreter.allocate_tensors()

indet = interpreter.get_input_details()[0]
outdet = interpreter.get_output_details()[0]

INDTYPE = indet['dtype']
OUTSCALE, OUTZERO = outdet['quantization']


def customnms(boxes, confidences, iouthresh=0.4):
    if len(boxes) == 0: return []
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = w * h
    order = confidences.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-5)
        inds = np.where(iou <= iouthresh)[0]
        order = order[inds + 1]
    return keep

def genframes():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        startt = time.time()
        succ, frame = cap.read()
        if not succ: break
        h, w, _ = frame.shape
        
        img = cv2.resize(frame, (IMGSIZE, IMGSIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if INDTYPE == np.int8:
            inp = (img.astype(np.int16) - 128).astype(np.int8)
        else:
            inp = img.astype(np.uint8)
            
        inp = np.expand_dims(inp, axis=0)
        
        interpreter.set_tensor(indet['index'], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(outdet['index'])[0]
        
        if OUTSCALE > 0:
            outf = (out.astype(np.float32) - OUTZERO) * OUTSCALE
        else:
            outf = out.astype(np.float32)

        confidences = outf[:, 4] * outf[:, 5]
        mask = confidences > 0.40
        
        currboxes = []

        if np.any(mask):
            vboxes = outf[mask]
            vconfs = confidences[mask]
            
            cx = vboxes[:, 0] * w
            cy = vboxes[:, 1] * h
            bw = vboxes[:, 2] * w
            bh = vboxes[:, 3] * h

            # int8 amnesia hack
            # quantization crushes decimals to zero so force a reticle so it doens crash opencv
            bw = np.maximum(bw, 150)
            bh = np.maximum(bh, 150)

            x1 = (cx - bw / 2).astype(int)
            y1 = (cy - bh / 2).astype(int)
            
            arr = np.column_stack((x1, y1, bw.astype(int), bh.astype(int)))
            idx = customnms(arr, vconfs)
            
            for i in idx:
                bx, by, bwv, bhv = arr[i]
                bx, by = max(0, bx), max(0, by)
                currboxes.append((bx, by, bwv, bhv, vconfs[i]))

        for bx, by, bwv, bhv, score in currboxes:
            cv2.rectangle(frame, (bx, by), (bx + bwv, by + bhv), (0, 255, 0), 2)
            cv2.putText(frame, f"OBJ {score:.2f}", (bx, max(20, by-10)), 0, 0.6, (0, 255, 0), 2)

        fps = 1.0 / (time.time() - startt)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 0, 0.8, (0, 255, 255), 2)
        
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/video_feed')
def videofeed(): return Response(genframes(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index(): return "<html><body style='background:#000;'><img src='/video_feed' style='width:100%'></body></html>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
