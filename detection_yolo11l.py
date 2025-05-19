import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from collections import defaultdict
import threading
import queue
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths for tracking
NEW_OBJECT_FILE = 'new.txt'
MISSING_OBJECT_FILE = 'missing.txt'

# Clear files at the start
open(NEW_OBJECT_FILE, 'w').close()
open(MISSING_OBJECT_FILE, 'w').close()

class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}  # Keeps full history
        self.frame_count = 0
        self.missing_frames_threshold = 10
        self.trajectories = defaultdict(list)
        self.reported_missing = set()

    def update(self, tracked_objects):
        self.frame_count += 1
        current_ids = set()
        new_objects = []
        missing_objects = []

        for track in tracked_objects:
            if not track.is_confirmed():
                continue

            track_id = str(track.track_id)
            box = track.to_tlbr()
            class_id = int(track.det_class) if hasattr(track, 'det_class') else 0
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            self.trajectories[track_id].append((center_x, center_y))

            # Update or add new track
            if track_id not in self.tracked_objects:
                new_objects.append((track_id, class_id))
                logger.info(f"New object detected: ID {track_id}, Class {class_id}")
                with open(NEW_OBJECT_FILE, 'a') as f:
                    f.write(f"{track_id}\n")

            self.tracked_objects[track_id] = {
                'box': box,
                'class_id': class_id,
                'last_seen': self.frame_count
            }

            current_ids.add(track_id)

        # Check for missing objects
        for track_id, obj_data in self.tracked_objects.items():
            if track_id not in current_ids:
                frames_missing = self.frame_count - obj_data['last_seen']
                if frames_missing >= self.missing_frames_threshold and track_id not in self.reported_missing:
                    missing_objects.append((track_id, obj_data['class_id']))
                    self.reported_missing.add(track_id)
                    logger.info(f"Object missing: ID {track_id}, Class {obj_data['class_id']}")
                    with open(MISSING_OBJECT_FILE, 'a') as f:
                        f.write(f"{track_id}\n")

        return new_objects, missing_objects

def read_frames(cap, frame_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    frame_queue.put(None)

def process_frames(frame_queue, output_queue, model, tracker, object_tracker, input_size):
    while True:
        frame = frame_queue.get()
        if frame is None:
            output_queue.put(None)
            break

        frame_resized = cv2.resize(frame, input_size)
        frame_yuv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)
        frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
        frame_processed = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)

        results = model(frame_processed, conf=0.5, iou=0.5)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf.cpu().numpy()
            cls = int(box.cls.cpu().numpy())
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame_processed)
        new_objects, missing_objects = object_tracker.update(tracks)

        for track in tracks:
            if not track.is_confirmed():
                continue
            box = track.to_tlbr().astype(int)
            cv2.rectangle(frame_processed, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            track_id = str(track.track_id)
            trajectory = object_tracker.trajectories[track_id]
            for i in range(1, len(trajectory)):
                cv2.line(frame_processed,
                         (int(trajectory[i - 1][0]), int(trajectory[i - 1][1])),
                         (int(trajectory[i][0]), int(trajectory[i][1])),
                         (255, 255, 0), 2)

        output_queue.put(frame_processed)

def write_frames(output_queue, out):
    while True:
        frame = output_queue.get()
        if frame is None:
            break
        out.write(frame)

def process_video(video_path, output_path, model_path='yolo11l.pt', input_size=(640, 640)):
    try:
        model = YOLO(model_path)
        model.export(format='engine', imgsz=input_size[0], device=0)
        model = YOLO(model_path.replace('.pt', '.engine'))
    except Exception as e:
        logger.error(f"Error loading or exporting YOLO model: {e}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 25)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, input_size)

    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    object_tracker = ObjectTracker()

    frame_queue = queue.Queue(maxsize=10)
    output_queue = queue.Queue(maxsize=10)

    read_thread = threading.Thread(target=read_frames, args=(cap, frame_queue))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, output_queue, model, tracker, object_tracker, input_size))
    write_thread = threading.Thread(target=write_frames, args=(output_queue, out))

    start_time = time.time()
    read_thread.start()
    process_thread.start()
    write_thread.start()

    read_thread.join()
    process_thread.join()
    write_thread.join()

    elapsed_time = time.time() - start_time
    achieved_fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) / elapsed_time if elapsed_time > 0 else 0

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return achieved_fps

if __name__ == '__main__':
    video_path = '/content/accident.mp4'
    output_path = '/content/video/output_video.mp4'
    fps = process_video(video_path, output_path)
    if fps is not None:
        print(f"Achieved FPS: {fps:.2f}")
