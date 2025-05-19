from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading
import queue
import time
import logging
from collections import defaultdict

# Configuration
YOLO_MODEL_PATH = 'yolo11l.pt'  # your model path
VIDEO_PATH = '/content/accident.mp4'
OUTPUT_PATH = '/content/video/output_video.mp4'
NEW_OBJECT_FILE = 'new.txt'
MISSING_OBJECT_FILE = 'missing.txt'

# Setup logging
logging.basicConfig(level=logging.INFO)
open(NEW_OBJECT_FILE, 'w').close()
open(MISSING_OBJECT_FILE, 'w').close()

class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.frame_count = 0
        self.trajectories = defaultdict(list)
        self.missing_frames_threshold = 10
        self.reported_missing = set()

    def update(self, tracks):
        self.frame_count += 1
        current_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = str(track.track_id)
            box = track.to_tlbr()
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            self.trajectories[tid].append((cx, cy))
            current_ids.add(tid)
            if tid not in self.tracked_objects:
                with open(NEW_OBJECT_FILE, 'a') as f:
                    f.write(tid + '\n')
                logging.info(f'New object: {tid}')
            self.tracked_objects[tid] = {'last_seen': self.frame_count}

        for tid, obj in list(self.tracked_objects.items()):
            if self.frame_count - obj['last_seen'] > self.missing_frames_threshold and tid not in self.reported_missing:
                with open(MISSING_OBJECT_FILE, 'a') as f:
                    f.write(tid + '\n')
                logging.info(f'Missing object: {tid}')
                self.reported_missing.add(tid)

def read_frames(cap, q):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        q.put(frame)
    q.put(None)

def process_frames(q_in, q_out, model, tracker, obj_tracker):
    while True:
        frame = q_in.get()
        if frame is None:
            q_out.put(None)
            break

        results = model.predict(source=frame, conf=0.4, iou=0.5, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)
        obj_tracker.update(tracks)

        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_tlbr().astype(int)
            tid = str(track.track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {tid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            for i in range(1, len(obj_tracker.trajectories[tid])):
                pt1 = tuple(map(int, obj_tracker.trajectories[tid][i - 1]))
                pt2 = tuple(map(int, obj_tracker.trajectories[tid][i]))
                cv2.line(frame, pt1, pt2, (255, 0, 255), 2)

        q_out.put(frame)

def write_output(q, writer):
    while True:
        frame = q.get()
        if frame is None:
            break
        writer.write(frame)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = max(int(cap.get(5)), 25)
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Load model on GPU (device=0)
    model = YOLO(YOLO_MODEL_PATH)
    model.to('cuda')  # move model to GPU

    tracker = DeepSort(max_age=20)
    obj_tracker = ObjectTracker()

    q_in = queue.Queue(maxsize=10)
    q_out = queue.Queue(maxsize=10)

    start = time.time()

    threads = [
        threading.Thread(target=read_frames, args=(cap, q_in)),
        threading.Thread(target=process_frames, args=(q_in, q_out, model, tracker, obj_tracker)),
        threading.Thread(target=write_output, args=(q_out, writer)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    cap.release()
    writer.release()
    end = time.time()
    logging.info(f"Done in {end - start:.2f} seconds")

if __name__ == '__main__':
    main()
