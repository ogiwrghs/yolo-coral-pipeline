import cv2
import numpy as np
import time
import threading
import logging
import queue
from collections import deque
from flask import Flask, Response
from scipy.optimize import linear_sum_assignment
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size

# --- Configuration Toggle ---
ENABLE_OPTICAL_FLOW = True  # Set to True for per-frame MOSSE+KLT updates; False to update only on inference frames

# --- Global Variables for Inference FPS ---
latest_inference_timestamp = None
inference_fps = 0.0

# --- Configure Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# --- Constants ---
MODEL_PATH = "v11n512.tflite" # specify which model to use
CONF_THRESHOLD = 0.4
INFERENCE_INTERVAL = 2  # Inference runs every Nth frame
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_DISPLAY_FPS = 30

# Tracker parameters
MAX_AGE = 2  # Maximum frames a tracker may go without update
IOU_THRESHOLD = 0.3
MAX_DETECTION_AREA = 0.5 * FRAME_WIDTH * FRAME_HEIGHT  # Skip detections covering >50% of view

# --- Global State ---
detection_lock = threading.Lock()
frame_lock = threading.Lock()

latest_detections = []     # Tracker boxes for display
frame_buffer = deque(maxlen=2)
frame_counter = 0
display_fps = 0.0          # For streaming FPS (when optical flow enabled)

# When optical flow is disabled, we use the latest processed (inference) frame.
latest_processed_frame = None

# Tracking state using MOSSE+KLT
trackers = dict()  # {id: CVTracker}
next_id = 1

# Inference queue and tracking of the latest inference frame processed
inference_queue = queue.Queue()
last_inference_frame = 0

# --- Utility Functions ---
def non_max_suppression(detections, iou_threshold=0.3):
    if len(detections) == 0:
        return []
    boxes = np.array([d[:4] for d in detections])
    scores = np.array([d[4] for d in detections])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return [detections[i] for i in keep]

def iou(a, b):
    a_x1, a_y1, a_x2, a_y2 = a[:4]
    b_x1, b_y1, b_x2, b_y2 = b[:4]
    xi1 = max(a_x1, b_x1)
    yi1 = max(a_y1, b_y1)
    xi2 = min(a_x2, b_x2)
    yi2 = min(a_y2, b_y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    union = a_area + b_area - inter + 1e-6
    return inter / union

def match_detections_to_trackers(detections, trackers_dict, iou_thresh=0.3):
    if not trackers_dict:
        return [], list(range(len(detections))), []
    tracker_items = list(trackers_dict.items())
    cost_matrix = np.zeros((len(detections), len(tracker_items)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, (tid, trk) in enumerate(tracker_items):
            tracker_box = trk.get_box()
            cost_matrix[d, t] = 1 - iou(det, tracker_box)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < 1 - iou_thresh:
            matches.append((i, j))
    unmatched_detections = [d for d in range(len(detections)) if d not in [m[0] for m in matches]]
    unmatched_trackers = [t for t in range(len(tracker_items)) if t not in [m[1] for m in matches]]
    return matches, unmatched_detections, unmatched_trackers

# --- OpenCV Tracker Wrapper (MOSSE + KLT) ---
try:
    tracker_create = cv2.TrackerMOSSE_create
except AttributeError:
    tracker_create = cv2.legacy.TrackerMOSSE_create

class CVTracker:
    def __init__(self, frame, bbox):
        self.tracker = tracker_create()
        x1, y1, x2, y2, score = bbox
        w = x2 - x1
        h = y2 - y1
        self.bbox = (x1, y1, w, h)
        self.tracker.init(frame, self.bbox)
        self.id = None
        self.score = score
        self.hits = 1
        self.time_since_update = 0
        self.age = 1
        # For KLT optical flow:
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.klt_points = self._init_klt_points(self.prev_gray)

    def _init_klt_points(self, gray):
        x, y, w, h = self.bbox
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=20, qualityLevel=0.3, minDistance=5)
        if pts is not None:
            pts[:, 0, 0] += x
            pts[:, 0, 1] += y
        return pts

    def update_with_optical_flow(self, curr_gray):
        if self.klt_points is None:
            return False
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.klt_points, None)
        if next_pts is None or status is None:
            return False
        try:
            good_old = np.array(self.klt_points)[status.flatten() == 1].reshape(-1, 2)
            good_new = np.array(next_pts)[status.flatten() == 1].reshape(-1, 2)
        except Exception as e:
            logger.error(f"Error reshaping KLT points: {e}")
            return False
        if good_old.shape[0] < 3 or good_new.shape[0] < 3:
            return False
        dx = np.mean(good_new[:, 0] - good_old[:, 0])
        dy = np.mean(good_new[:, 1] - good_old[:, 1])
        x, y, w, h = self.bbox
        self.bbox = (int(x + dx), int(y + dy), w, h)
        self.klt_points = good_new.reshape(-1, 1, 2)
        self.prev_gray = curr_gray.copy()
        self.age += 1
        self.time_since_update += 1
        return True

    def update(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            x, y, w, h = bbox
            self.bbox = (int(x), int(y), int(w), int(h))
        self.age += 1
        self.time_since_update += 1
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return success, self.get_box()

    def get_box(self):
        x, y, w, h = self.bbox
        return [x, y, x + w, y + h, self.score]

    def reinitialize(self, frame, bbox):
        x1, y1, x2, y2, score = bbox
        w = x2 - x1
        h = y2 - y1
        self.bbox = (x1, y1, w, h)
        self.tracker = tracker_create()
        self.tracker.init(frame, self.bbox)
        self.time_since_update = 0
        self.hits += 1
        self.score = 0.7 * self.score + 0.3 * score
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.klt_points = self._init_klt_points(self.prev_gray)

# --- Camera Capture Thread ---
def capture_frames():
    global frame_counter
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    logger.info("Camera stream started")
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                frame_counter += 1
                frame_buffer.append(frame.copy())
                # Push a copy into the inference queue on every INFERENCE_INTERVAL frame
                if frame_counter % INFERENCE_INTERVAL == 0:
                    inference_queue.put((frame_counter, frame.copy()))
        else:
            logger.warning("Failed to capture frame, retrying...")
            time.sleep(0.001)

# --- Processing Thread (Optical Flow Updates) ---
def process_frames():
    prev_frame_gray = None
    while True:
        if not frame_buffer:
            time.sleep(0.001)
            continue
        with frame_lock:
            frame = frame_buffer[-1].copy()
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ENABLE_OPTICAL_FLOW:
            with detection_lock:
                for trk in list(trackers.values()):
                    trk.update_with_optical_flow(curr_gray)
            with detection_lock:
                lost_ids = []
                for tid, trk in list(trackers.items()):
                    success, _ = trk.update(frame)
                    if not success and trk.time_since_update > MAX_AGE:
                        lost_ids.append(tid)
                for tid in lost_ids:
                    del trackers[tid]
        prev_frame_gray = curr_gray.copy()
        time.sleep(0.01)

# --- Inference Worker Threads ---
def inference_worker(worker_id):
    global next_id, latest_processed_frame, latest_inference_timestamp, inference_fps, last_inference_frame
    try:
        interpreter = make_interpreter(MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        model_w, model_h = input_size(interpreter)
        scale_x = FRAME_WIDTH / model_w
        scale_y = FRAME_HEIGHT / model_h
        logger.info(f"Inference worker {worker_id} initialized: input size {model_w}x{model_h}")
    except Exception as e:
        logger.error(f"Inference worker {worker_id} failed to initialize model: {str(e)}")
        return

    while True:
        try:
            frame_info = inference_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        frame_num, frame = frame_info
        # Only process if this frame is newer than what was already processed
        with detection_lock:
            if frame_num <= last_inference_frame:
                inference_queue.task_done()
                continue

        start_time = time.perf_counter()
        resized = cv2.resize(frame, (model_w, model_h))
        input_data = np.expand_dims(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, axis=0)
        try:
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0].T
        except Exception as e:
            logger.error(f"Inference worker {worker_id} inference failed: {str(e)}")
            inference_queue.task_done()
            continue

        scores = output[:, 4]
        valid = scores > CONF_THRESHOLD
        boxes = output[valid]
        detections = []
        for box in boxes:
            cx = box[0] * model_w
            cy = box[1] * model_h
            w_box = box[2] * model_w
            h_box = box[3] * model_h
            x1 = int((cx - w_box / 2) * scale_x)
            y1 = int((cy - h_box / 2) * scale_y)
            x2 = int((cx + w_box / 2) * scale_x)
            y2 = int((cy + h_box / 2) * scale_y)
            x1 = max(0, min(x1, FRAME_WIDTH))
            y1 = max(0, min(y1, FRAME_HEIGHT))
            x2 = max(0, min(x2, FRAME_WIDTH))
            y2 = max(0, min(y2, FRAME_HEIGHT))
            area = (x2 - x1) * (y2 - y1)
            if area > MAX_DETECTION_AREA:
                continue
            detections.append([x1, y1, x2, y2, float(box[4])])
        detections = non_max_suppression(detections, iou_threshold=IOU_THRESHOLD)
        with detection_lock:
            if frame_num > last_inference_frame:
                last_inference_frame = frame_num
                matches, unmatched_dets, unmatched_trks = match_detections_to_trackers(detections, trackers, iou_thresh=IOU_THRESHOLD)
                tracker_keys = list(trackers.keys())
                for m in matches:
                    tid = tracker_keys[m[1]]
                    trackers[tid].reinitialize(frame, detections[m[0]])
                for d in unmatched_dets:
                    tid = next_id
                    next_id += 1
                    new_tracker = CVTracker(frame, detections[d])
                    new_tracker.id = tid
                    trackers[tid] = new_tracker
                for idx in unmatched_trks:
                    tid = tracker_keys[idx]
                    if trackers[tid].time_since_update > MAX_AGE:
                        del trackers[tid]
                inference_duration = (time.perf_counter() - start_time) * 1000
                current_inference_timestamp = time.time()
                if latest_inference_timestamp is not None:
                    delta = current_inference_timestamp - latest_inference_timestamp
                    if delta > 0:
                        inference_fps = 1.0 / delta
                else:
                    inference_fps = 0.0
                latest_inference_timestamp = current_inference_timestamp
                latest_processed_frame = frame.copy()
                logger.debug(f"Inference worker {worker_id} time: {inference_duration:.1f}ms, "
                             f"Detections: {len(detections)}, "
                             f"Active trackers: {len(trackers)}")
        inference_queue.task_done()

# --- Flask Application ---
app = Flask(__name__)

def generate_frames():
    global display_fps, inference_fps
    last_time = time.time()
    last_frame = None
    frame_timer = time.time()
    while True:
        try:
            sleep_time = 1.0 / MAX_DISPLAY_FPS - (time.time() - frame_timer)
            if sleep_time > 0:
                time.sleep(sleep_time)
            frame_timer = time.time()
            with frame_lock:
                if not ENABLE_OPTICAL_FLOW and latest_processed_frame is not None:
                    frame = latest_processed_frame.copy()
                elif frame_buffer:
                    frame = frame_buffer[-1].copy()
                else:
                    frame = last_frame if last_frame is not None else np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            with detection_lock:
                boxes = latest_detections.copy()
                # Also update with current tracker boxes for display
                boxes = [trk.get_box() for trk in trackers.values()]
            box_color = (0, 255, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for box in boxes:
                try:
                    x1, y1, x2, y2, score = box
                    x1 = int(np.clip(x1, 0, FRAME_WIDTH))
                    y1 = int(np.clip(y1, 0, FRAME_HEIGHT))
                    x2 = int(np.clip(x2, 0, FRAME_WIDTH))
                    y2 = int(np.clip(y2, 0, FRAME_HEIGHT))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    label = f"{score:.2f}"
                    label_size, _ = cv2.getTextSize(label, font, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), box_color, -1)
                    cv2.putText(frame, label, (x1, y1-10), font, 0.6, (0, 0, 0), 2)
                except Exception as e:
                    logger.error(f"Error drawing box: {str(e)}")
                    continue
            now = time.time()
            dt = now - last_time
            if dt > 0:
                if ENABLE_OPTICAL_FLOW:
                    display_fps = 0.2 * (1 / dt) + 0.8 * display_fps
                    fps_text = f"Display FPS: {display_fps:.1f}"
                else:
                    fps_text = f"Display FPS: {inference_fps:.1f}"
            last_time = now
            cv2.putText(frame, fps_text, (10, 30), font, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Trackers: {len(boxes)}", (10, 60), font, 0.8, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            last_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return f'''
    <html>
      <head>
        <title>Object Detection with MOSSE + KLT Tracking</title>
        <style>
          body {{ font-family: Arial, sans-serif; text-align: center; margin: 0; padding: 20px; }}
          h1 {{ color: #333; }}
          .container {{ max-width: 800px; margin: 0 auto; }}
          img {{ border: 1px solid #ddd; max-width: 100%; height: auto; }}
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Object Detection with MOSSE + KLT Tracking</h1>
          <p>Optical Flow Enabled: {ENABLE_OPTICAL_FLOW}</p>
          <img src="/video_feed" width="640" height="480">
        </div>
      </body>
    </html>
    '''

if __name__ == '__main__':
    # Start camera capture thread
    threading.Thread(target=capture_frames, daemon=True).start()
    logger.info("Started capture thread")
    # Start optical flow processing thread
    threading.Thread(target=process_frames, daemon=True).start()
    logger.info("Started processing thread")
    # Start two inference worker threads
    threading.Thread(target=inference_worker, args=(1,), daemon=True).start()
    threading.Thread(target=inference_worker, args=(2,), daemon=True).start()
    logger.info("Started two inference threads")
    logger.info("Starting web server on port 5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
