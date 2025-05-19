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

# Configure logging for alerts
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectTracker:
    def __init__(self, missing_frames_threshold=10):
        self.tracked_objects = {}
        self.frame_count = 0
        self.missing_frames_threshold = missing_frames_threshold
        self.trajectories = defaultdict(list)

    def update(self, tracked_objects):
        self.frame_count += 1
        current_objects = {}
        new_objects = []
        missing_objects = []

        for track in tracked_objects:
            if not track.is_confirmed():
                continue
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

        objects_to_remove = []
        for track_id, obj_data in self.tracked_objects.items():
            if track_id not in current_objects:
                if self.frame_count - obj_data['last_seen'] >= self.missing_frames_threshold:
                    missing_objects.append((track_id, obj_data['class_id']))
                    logger.info(f"Object missing: ID {track_id}, Class {obj_data['class_id']}")
                    objects_to_remove.append(track_id)

        for track_id in objects_to_remove:
            del self.tracked_objects[track_id]

        self.tracked_objects.update(current_objects)
        return new_objects, missing_objects

def read_frames(cap, frame_queue, frame_skip=0):
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_skip == 0 or frame_count % (frame_skip + 1) == 0:
            frame_queue.put(frame)
        frame_count += 1
    frame_queue.put(None)

def process_frames(frame_queue, output_queue, model, tracker, object_tracker,
                   confidence_threshold=0.5, iou_threshold=0.5, yolo_imgsz=640):
    while True:
        frame = frame_queue.get()
        if frame is None:
            output_queue.put(None)
            break

        resized_frame = cv2.resize(frame, (yolo_imgsz, yolo_imgsz))

        results = model(resized_frame, conf=confidence_threshold, iou=iou_threshold, verbose=False)

        detections = []
        original_height, original_width = frame.shape[:2]
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf.cpu().numpy()
            cls = int(box.cls.cpu().numpy())

            # Scale the bounding box back to the original frame size
            x1 = int(x1 * original_width / yolo_imgsz)
            y1 = int(y1 * original_height / yolo_imgsz)
            x2 = int(x2 * original_width / yolo_imgsz)
            y2 = int(y2 * original_height / yolo_imgsz)

            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)
        new_objects, missing_objects = object_tracker.update(tracks)

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

        for _, class_id in new_objects:
            cv2.putText(frame, f"New Object {class_id}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for _, class_id in missing_objects:
            cv2.putText(frame, f"Missing Object {class_id}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        output_queue.put(frame)

def write_frames(output_queue, out):
    while True:
        frame = output_queue.get()
        if frame is None:
            break
        out.write(frame)

def process_video(video_path, output_path, model_path='/content/yolo11n.pt',
                  confidence_threshold=0.5, iou_threshold=0.5, missing_frames_threshold=10,
                  export_engine=True, yolo_imgsz=640, frame_skip=0):
    model = None
    try:
        model = YOLO(model_path)
        if export_engine:
            try:
                model.export(format='engine', imgsz=yolo_imgsz, device=0)
                model = YOLO('/content/yolo11n.engine')
                logger.info(f"YOLO model exported to TensorRT engine with size {yolo_imgsz}.")
                # Force a warmup with the correct input size after loading the engine
                dummy_input = np.zeros((1, 3, yolo_imgsz, yolo_imgsz), dtype=np.float32)
                _ = model(dummy_input)
            except Exception as e_engine:
                logger.warning(f"Error exporting to TensorRT engine: {e_engine}. Falling back to ONNX.")
                try:
                    model.export(format='onnx', imgsz=yolo_imgsz)
                    model = YOLO('yolo11n.onnx')
                    logger.info(f"YOLO model exported to ONNX with size {yolo_imgsz}.")
                except Exception as e_onnx:
                    logger.error(f"Error exporting to ONNX: {e_onnx}. Using PyTorch model.")
                    model = YOLO(model_path)
        else:
            # Warmup for PyTorch model with the target size
            dummy_input = np.zeros((1, 3, yolo_imgsz, yolo_imgsz), dtype=np.float32)
            _ = model(dummy_input)

    except Exception as e_load:
        logger.error(f"Error loading YOLO model: {e_load}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    object_tracker = ObjectTracker(missing_frames_threshold=missing_frames_threshold)

    frame_queue = queue.Queue(maxsize=10)
    output_queue = queue.Queue(maxsize=10)

    read_thread = threading.Thread(target=read_frames, args=(cap, frame_queue, frame_skip))
    process_thread = threading.Thread(target=process_frames,
                                      args=(frame_queue, output_queue, model, tracker, object_tracker,
                                            confidence_threshold, iou_threshold, yolo_imgsz))
    write_thread = threading.Thread(target=write_frames, args=(output_queue, out))

    start_time = time.time()
    read_thread.start()
    process_thread.start()
    write_thread.start()

    read_thread.join()
    process_thread.join()
    write_thread.join()

    elapsed_time = time.time() - start_time
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frame_count = total_frames if frame_skip == 0 else total_frames // (frame_skip + 1)
    achieved_fps = processed_frame_count / elapsed_time if elapsed_time > 0 else 0

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return achieved_fps

if __name__ == '__main__':
    video_path = '/content/accident.mp4'
    output_path = '/content/video/output_video.mp4'
    achieved_fps = process_video(video_path, output_path,
                                 confidence_threshold=0.4,
                                 iou_threshold=0.4,
                                 missing_frames_threshold=15,
                                 export_engine=True,
                                 yolo_imgsz=480,
                                 frame_skip=1)

    if achieved_fps is not None:
        print(f"Achieved FPS: {achieved_fps:.2f}")