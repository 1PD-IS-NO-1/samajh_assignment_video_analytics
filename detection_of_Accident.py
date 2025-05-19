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
import pygame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Pygame mixer for audio
pygame.mixer.init()
emergency_sound = pygame.mixer.Sound('emergency.mp3')  # Ensure emergency.mp3 is available

class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.frame_count = 0
        self.missing_frames_threshold = 10
        self.trajectories = defaultdict(list)
        self.previous_objects = {}

    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    def detect_collisions(self):
        collisions = []
        for id1 in self.tracked_objects:
            for id2 in self.tracked_objects:
                if id1 < id2:
                    box1 = self.tracked_objects[id1]['box']
                    box2 = self.tracked_objects[id2]['box']
                    if self.calculate_iou(box1, box2) > 0.1:
                        if id1 in self.previous_objects and id2 in self.previous_objects:
                            prev_box1 = self.previous_objects[id1]['box']
                            prev_box2 = self.previous_objects[id2]['box']
                            if self.calculate_iou(prev_box1, prev_box2) == 0:
                                # Calculate centers and velocities
                                center1_curr = ((box1[0] + box1[2])/2, (box1[1] + box1[3])/2)
                                center2_curr = ((box2[0] + box2[2])/2, (box2[1] + box2[3])/2)
                                prev_center1 = ((prev_box1[0] + prev_box1[2])/2, (prev_box1[1] + prev_box1[3])/2)
                                prev_center2 = ((prev_box2[0] + prev_box2[2])/2, (prev_box2[1] + prev_box2[3])/2)
                                V1 = (center1_curr[0] - prev_center1[0], center1_curr[1] - prev_center1[1])
                                V2 = (center2_curr[0] - prev_center2[0], center2_curr[1] - prev_center2[1])
                                relative_V = (V2[0] - V1[0], V2[1] - V1[1])
                                relative_pos = (center2_curr[0] - center1_curr[0], center2_curr[1] - center1_curr[1])
                                dot_product = relative_pos[0] * relative_V[0] + relative_pos[1] * relative_V[1]
                                relative_speed = (relative_V[0]**2 + relative_V[1]**2)**0.5
                                if dot_product < 0 and relative_speed > 5:
                                    collisions.append((id1, id2))
        return collisions

    def update(self, tracked_objects):
        self.frame_count += 1
        self.previous_objects = {k: v.copy() for k, v in self.tracked_objects.items()}
        current_objects = {}
        new_objects = []
        missing_objects = []

        for track in tracked_objects:
            box = track.to_tlbr()
            track_id = track.track_id
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

        for track_id, obj_data in list(self.tracked_objects.items()):
            if track_id not in current_objects:
                if self.frame_count - obj_data['last_seen'] >= self.missing_frames_threshold:
                    missing_objects.append((track_id, obj_data['class_id']))
                    logger.info(f"Object missing: ID {track_id}, Class {obj_data['class_id']}")

        self.tracked_objects = current_objects
        collisions = self.detect_collisions()
        return new_objects, missing_objects, collisions

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
        new_objects, missing_objects, collisions = object_tracker.update(tracks)

        if collisions:
            for collision in collisions:
                logger.info(f"Collision detected between {collision[0]} and {collision[1]}")
                emergency_sound.play()

        if draw:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                box = track.to_tlbr().astype(int)
                track_id = track.track_id
                if any(track_id in collision for collision in collisions):
                    color = (0, 0, 255)  # Red for collision
                else:
                    color = (0, 255, 0)  # Green for normal
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                trajectory = object_tracker.trajectories.get(track_id, [])
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
        model(np.zeros((640, 640, 3), dtype=np.uint8))  # Warm-up
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
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, output_queue, model, tracker, object_tracker, (416, 416), True))
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