"""
Human detection module using YOLOv8-nano.
Detects persons in frames and tracks them across frames using centroid matching.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
import config


class CentroidTracker:
    """
    Simple centroid-based object tracker.
    Assigns persistent IDs to detected persons across frames
    by matching centroids between consecutive frames.
    """

    def __init__(self, max_disappeared=None, max_distance=None):
        self.max_disappeared = max_disappeared or config.TRACKER_MAX_DISAPPEARED
        self.max_distance = max_distance or config.TRACKER_MAX_DISTANCE
        self.next_id = 0
        self.objects = OrderedDict()       # id -> centroid (cx, cy)
        self.bboxes = OrderedDict()        # id -> (x1, y1, x2, y2)
        self.disappeared = OrderedDict()   # id -> frame count since last seen

    def _register(self, centroid, bbox):
        """Register a new object with a unique ID."""
        self.objects[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, object_id):
        """Remove a tracked object."""
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections: list of (x1, y1, x2, y2, confidence) tuples

        Returns:
            dict of {id: {"bbox": (x1,y1,x2,y2), "centroid": (cx,cy), "conf": float}}
        """
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self._build_result(detections)

        # Compute centroids of new detections
        input_centroids = []
        input_bboxes = []
        input_confs = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            conf = det[4] if len(det) > 4 else 1.0
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_centroids.append((cx, cy))
            input_bboxes.append((x1, y1, x2, y2))
            input_confs.append(conf)

        input_centroids = np.array(input_centroids)

        # If no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], input_bboxes[i])
            return self._build_result(detections)

        # Match existing objects to new detections using distance
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(object_centroids, input_centroids)

        # Hungarian-like greedy matching (simple approach)
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.bboxes[object_id] = input_bboxes[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects (disappeared)
        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self._deregister(object_id)

        # Handle unmatched new detections (register)
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self._register(input_centroids[col], input_bboxes[col])

        return self._build_result(detections)

    def _build_result(self, detections):
        """Build the result dict with all tracked objects."""
        result = {}
        for oid in self.objects:
            result[oid] = {
                "bbox": self.bboxes[oid],
                "centroid": tuple(self.objects[oid]),
            }
        return result


class PersonDetector:
    """
    YOLOv8-based person detector.
    Detects persons and maintains tracking IDs across frames.
    """

    def __init__(self):
        print("[Detector] Loading YOLOv8 model...")
        self.model = YOLO(config.YOLO_MODEL)
        self.tracker = CentroidTracker()
        self.person_class = config.YOLO_PERSON_CLASS
        self.confidence = config.YOLO_CONFIDENCE
        print("[Detector] Model loaded successfully")

    def detect(self, frame):
        """
        Detect persons in a frame.

        Args:
            frame: BGR numpy array

        Returns:
            tracked_persons: dict of {id: {"bbox": (x1,y1,x2,y2), "centroid": (cx,cy)}}
            raw_detections: list of (x1, y1, x2, y2, confidence) for this frame
        """
        # Run YOLOv8 inference
        results = self.model(frame, verbose=False, conf=self.confidence)

        raw_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                if cls != self.person_class:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                raw_detections.append((int(x1), int(y1), int(x2), int(y2), conf))

        # Update tracker
        tracked = self.tracker.update(raw_detections)

        return tracked, raw_detections

    def draw_detections(self, frame, tracked_persons):
        """
        Draw bounding boxes and IDs on the frame.

        Args:
            frame: BGR numpy array
            tracked_persons: dict from detect()

        Returns:
            annotated frame
        """
        for person_id, data in tracked_persons.items():
            x1, y1, x2, y2 = [int(v) for v in data["bbox"]]
            cx, cy = [int(v) for v in data["centroid"]]

            # Color based on ID (consistent per person)
            color_hue = (person_id * 47) % 180
            color = cv2.cvtColor(
                np.uint8([[[color_hue, 200, 255]]]),
                cv2.COLOR_HSV2BGR
            )[0][0].tolist()

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw ID label with background
            label = f"Person {person_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Draw centroid dot
            cv2.circle(frame, (cx, cy), 4, color, -1)

        # Person count
        count = len(tracked_persons)
        if count > 0:
            cv2.putText(frame, f"Persons: {count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame
