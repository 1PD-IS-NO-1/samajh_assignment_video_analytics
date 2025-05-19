# Fpas will be around 20-25
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

class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.frame_count = 0
        self.missing_frames_threshold = 10
        self.trajectories = defaultdict(list)

    def update(self, tracked_objects):
        self.frame_count += 1
        current_objects = {}
        new_objects = []
        missing_objects = []

        for track in tracked_objects:
            box = track.to_tlbr()
            track_id = str(track.track_id)
            class_id = int(track.det_class) if hasattr(track, 'det_class') else 0
            current_objects[track_id] = {
                'box': box,
                'class_id': class_id,
                'last_seen': self.frame_count
            }
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            self.trajectories[track_id].append((center_x, center_y))

        for track_id, obj_data in current_objects.items():
            if track_id not in self.tracked_objects:
                new_objects.append((track_id, obj_data['class_id']))
                logger.info(f"New object detected: ID {track_id}, Class {obj_data['class_id']}")

        for track_id, obj_data in self.tracked_objects.items():
            if track_id not in current_objects:
                if self.frame_count - obj_data['last_seen'] >= self.missing_frames_threshold:
                    missing_objects.append((track_id, obj_data['class_id']))
                    logger.info(f"Object missing: ID {track_id}, Class {obj_data['class_id']}")

        self.tracked_objects = current_objects
        return new_objects, missing_objects

def read_frames(cap, frame_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    frame_queue.put(None)

def process_frames(frame_queue, output_queue, model, tracker, object_tracker, resize=(416, 416), draw=False):
    while True:
        frame = frame_queue.get()
        if frame is None:
            output_queue.put(None)
            break

        frame = cv2.resize(frame, resize)

        results = model(frame, conf=0.5, iou=0.5)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf.cpu().numpy()
            cls = int(box.cls.cpu().numpy())
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)
        new_objects, missing_objects = object_tracker.update(tracks)

        if draw:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                box = track.to_tlbr().astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                track_id = track.track_id
                trajectory = object_tracker.trajectories[track_id]
                for i in range(1, len(trajectory)):
                    cv2.line(frame,
                             (int(trajectory[i - 1][0]), int(trajectory[i - 1][1])),
                             (int(trajectory[i][0]), int(trajectory[i][1])),
                             (255, 255, 0), 2)

        output_queue.put(frame)

def write_frames(output_queue, out, write_video):
    while True:
        frame = output_queue.get()
        if frame is None:
            break
        if write_video:
            out.write(frame)

def process_video(video_path, output_path, model_path='yolov8n.pt', write_video=False):
    try:
        model = YOLO(model_path)
        model(np.zeros((640, 640, 3), dtype=np.uint8))  # warm-up
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (416, 416))

    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    object_tracker = ObjectTracker()

    frame_queue = queue.Queue(maxsize=10)
    output_queue = queue.Queue(maxsize=10)

    read_thread = threading.Thread(target=read_frames, args=(cap, frame_queue))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, output_queue, model, tracker, object_tracker, (416, 416), False))
    write_thread = threading.Thread(target=write_frames, args=(output_queue, out, write_video))

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
    video_path = '/content/video/person-bicycle-car-detection.mp4'
    output_path = '/content/video/output_video.mp4'
    fps = process_video(video_path, output_path, model_path='yolov8n.pt', write_video=False)
    if fps is not None:
        print(f"Achieved FPS: {fps:.2f}")