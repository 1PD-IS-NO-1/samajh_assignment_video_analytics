#fps will be around more than 50
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import uuid
import torch
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectTracker:
    def __init__(self, iou_threshold=0.5):
        self.tracked_objects = {}
        self.history = defaultdict(list)
        self.iou_threshold = iou_threshold
        self.missing_frames_threshold = 10
        self.frame_count = 0

    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        return inter_area / (box1_area + box2_area - inter_area)

    def update(self, detections):
        self.frame_count += 1
        current_objects = {}
        new_objects = []
        missing_objects = []

        for det in detections:
            box = det[:4]
            class_id = int(det[5])
            matched = False
            best_iou = 0
            best_id = None

            for obj_id, obj_data in self.tracked_objects.items():
                iou_score = self.iou(box, obj_data['box'])
                if iou_score > best_iou and iou_score > self.iou_threshold:
                    best_iou = iou_score
                    best_id = obj_id
                    matched = True

            if matched:
                self.tracked_objects[best_id]['box'] = box
                self.tracked_objects[best_id]['last_seen'] = self.frame_count
                current_objects[best_id] = self.tracked_objects[best_id]
            else:
                new_id = str(uuid.uuid4())
                current_objects[new_id] = {'box': box, 'class_id': class_id, 'last_seen': self.frame_count}
                new_objects.append((new_id, class_id))

        for obj_id, obj_data in self.tracked_objects.items():
            if obj_id not in current_objects:
                if self.frame_count - obj_data['last_seen'] >= self.missing_frames_threshold:
                    missing_objects.append((obj_id, obj_data['class_id']))

        self.tracked_objects = current_objects
        return new_objects, missing_objects

def process_video(video_path, output_path, model_path='yolov8n.pt'):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file {video_path}")
        return None

    tracker = ObjectTracker()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Try mp4v codec, fallback to XVID if needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        logger.warning("mp4v codec failed, trying XVID")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace('.mp4', '.avi')  # Change extension for XVID
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        logger.error(f"Error: Could not initialize VideoWriter for {output_path}")
        cap.release()
        return None

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5, iou=0.5)
        detections = results[0].boxes.data.cpu().numpy()

        new_objects, missing_objects = tracker.update(detections)

        for det in detections:
            x, y, w, h = det[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for obj_id, class_id in new_objects:
            cv2.putText(frame, f"New Object {class_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for obj_id, class_id in missing_objects:
            cv2.putText(frame, f"Missing Object {class_id}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(frame)
        frame_count += 1
        logger.debug(f"Processed frame {frame_count}")

    elapsed_time = time.time() - start_time
    achieved_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Verify output file
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(f"Output video saved successfully at {output_path}")
    else:
        logger.error(f"Output video not found or empty at {output_path}")

    return achieved_fps

if __name__ == '__main__':
    video_path = '/content/accident.mp4'
    output_path = '/content/video/output_video.mp4'
    fps = process_video(video_path, output_path)
    if fps is not None:
        print(f"Achieved FPS: {fps:.2f}")