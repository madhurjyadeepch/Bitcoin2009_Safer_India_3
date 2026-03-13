"""
Pose estimation module using MediaPipe Pose Landmarker (Tasks API).
Extracts 33 body landmarks per detected person, calculates joint angles,
body tilt, and limb velocities for anomaly detection.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from collections import defaultdict, deque
import os
import config


# MediaPipe landmark indices (key ones for anomaly detection)
LANDMARK = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Skeleton connections for drawing
POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


class PoseEstimator:
    """
    MediaPipe Pose Landmarker (Tasks API) wrapper.
    Processes cropped person images and extracts body landmarks with
    derived features (angles, velocities) for anomaly detection.
    """

    def __init__(self):
        print("[PoseEstimator] Initializing MediaPipe Pose Landmarker...")

        # Resolve model path
        model_path = getattr(config, "POSE_MODEL_PATH", "pose_landmarker_lite.task")
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose model not found at: {model_path}\n"
                "Download it with:\n"
                "wget -O pose_landmarker_lite.task "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            )

        # Create PoseLandmarker with IMAGE mode (we process individual person crops)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,  # one pose per crop (each crop is a single person)
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # History buffer for velocity calculation: {person_id: deque of landmark dicts}
        self._history = defaultdict(lambda: deque(maxlen=config.FIGHT_FRAME_WINDOW))
        # Track which IDs are active to clean up stale entries
        self._active_ids = set()
        print("[PoseEstimator] Ready")

    def estimate(self, frame, tracked_persons):
        """
        Run pose estimation on each tracked person.

        Args:
            frame: BGR numpy array (full frame)
            tracked_persons: dict {id: {"bbox": (x1,y1,x2,y2), "centroid": (cx,cy)}}

        Returns:
            poses: dict {person_id: {
                "landmarks": {name: (x, y)} in frame coordinates,
                "angles": {"left_elbow": float, "right_elbow": float, ...},
                "body_tilt": float (degrees from vertical),
                "velocities": {"left_wrist": float, "right_wrist": float, ...} px/frame,
                "bbox": (x1, y1, x2, y2)
            }}
        """
        h, w = frame.shape[:2]
        poses = {}
        current_ids = set()

        for person_id, data in tracked_persons.items():
            x1, y1, x2, y2 = [int(v) for v in data["bbox"]]
            current_ids.add(person_id)

            # Ensure bbox is within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < 20 or crop_h < 20:
                continue

            # Crop and process
            person_crop = frame[y1:y2, x1:x2]
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image from numpy array
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

            # Run pose detection
            try:
                result = self.landmarker.detect(mp_image)
            except Exception:
                continue

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                continue

            # Get the first (and only) detected pose
            pose_landmarks = result.pose_landmarks[0]

            # Convert landmarks to frame coordinates
            landmarks = {}
            for name, idx in LANDMARK.items():
                lm = pose_landmarks[idx]
                # Convert from normalized crop coords to frame coords
                px = int(lm.x * crop_w + x1)
                py = int(lm.y * crop_h + y1)
                landmarks[name] = (px, py)

            # Calculate joint angles
            angles = self._calculate_angles(landmarks)

            # Calculate body tilt
            body_tilt = self._calculate_body_tilt(landmarks)

            # Store in history and calculate velocities
            self._history[person_id].append(landmarks)
            velocities = self._calculate_velocities(person_id)

            poses[person_id] = {
                "landmarks": landmarks,
                "angles": angles,
                "body_tilt": body_tilt,
                "velocities": velocities,
                "bbox": (x1, y1, x2, y2),
            }

        # Clean up history for persons no longer tracked
        stale_ids = self._active_ids - current_ids
        for sid in stale_ids:
            if sid in self._history:
                del self._history[sid]
        self._active_ids = current_ids

        return poses

    def _calculate_angle(self, a, b, c):
        """
        Calculate angle at point b given three points (a, b, c).
        Returns angle in degrees.
        """
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)

        dot = np.dot(ba, bc)
        mag_ba = np.linalg.norm(ba)
        mag_bc = np.linalg.norm(bc)

        if mag_ba < 1e-6 or mag_bc < 1e-6:
            return 0.0

        cos_angle = np.clip(dot / (mag_ba * mag_bc), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def _calculate_angles(self, landmarks):
        """Calculate key joint angles."""
        angles = {}

        # Elbow angles
        if all(k in landmarks for k in ("left_shoulder", "left_elbow", "left_wrist")):
            angles["left_elbow"] = self._calculate_angle(
                landmarks["left_shoulder"],
                landmarks["left_elbow"],
                landmarks["left_wrist"],
            )

        if all(k in landmarks for k in ("right_shoulder", "right_elbow", "right_wrist")):
            angles["right_elbow"] = self._calculate_angle(
                landmarks["right_shoulder"],
                landmarks["right_elbow"],
                landmarks["right_wrist"],
            )

        # Knee angles
        if all(k in landmarks for k in ("left_hip", "left_knee", "left_ankle")):
            angles["left_knee"] = self._calculate_angle(
                landmarks["left_hip"],
                landmarks["left_knee"],
                landmarks["left_ankle"],
            )

        if all(k in landmarks for k in ("right_hip", "right_knee", "right_ankle")):
            angles["right_knee"] = self._calculate_angle(
                landmarks["right_hip"],
                landmarks["right_knee"],
                landmarks["right_ankle"],
            )

        # Shoulder angles (arm extension)
        if all(k in landmarks for k in ("left_elbow", "left_shoulder", "left_hip")):
            angles["left_shoulder"] = self._calculate_angle(
                landmarks["left_elbow"],
                landmarks["left_shoulder"],
                landmarks["left_hip"],
            )

        if all(k in landmarks for k in ("right_elbow", "right_shoulder", "right_hip")):
            angles["right_shoulder"] = self._calculate_angle(
                landmarks["right_elbow"],
                landmarks["right_shoulder"],
                landmarks["right_hip"],
            )

        return angles

    def _calculate_body_tilt(self, landmarks):
        """
        Calculate body tilt angle from vertical.
        Uses the midpoint of shoulders and midpoint of hips to form the torso line.
        Returns angle in degrees (0 = upright, 90 = horizontal).
        """
        required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if not all(k in landmarks for k in required):
            return 0.0

        # Midpoints
        mid_shoulder = (
            (landmarks["left_shoulder"][0] + landmarks["right_shoulder"][0]) / 2,
            (landmarks["left_shoulder"][1] + landmarks["right_shoulder"][1]) / 2,
        )
        mid_hip = (
            (landmarks["left_hip"][0] + landmarks["right_hip"][0]) / 2,
            (landmarks["left_hip"][1] + landmarks["right_hip"][1]) / 2,
        )

        # Vector from hip to shoulder (torso direction)
        dx = mid_shoulder[0] - mid_hip[0]
        dy = mid_shoulder[1] - mid_hip[1]  # y-axis is inverted in image coords

        # Angle from vertical
        torso_length = np.sqrt(dx * dx + dy * dy)
        if torso_length < 1e-6:
            return 0.0

        # Angle between torso vector and vertical (0, -1)
        cos_angle = np.clip(-dy / torso_length, -1.0, 1.0)  # -dy because y is inverted
        return float(np.degrees(np.arccos(cos_angle)))

    def _calculate_velocities(self, person_id):
        """
        Calculate per-frame velocities for key limb points.
        Uses the last two frames in history.
        Returns velocities in pixels/frame.
        """
        velocities = {}
        history = self._history[person_id]

        if len(history) < 2:
            return {
                "left_wrist": 0.0, "right_wrist": 0.0,
                "left_ankle": 0.0, "right_ankle": 0.0,
            }

        prev = history[-2]
        curr = history[-1]

        for joint in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
            if joint in prev and joint in curr:
                dx = curr[joint][0] - prev[joint][0]
                dy = curr[joint][1] - prev[joint][1]
                velocities[joint] = float(np.sqrt(dx * dx + dy * dy))
            else:
                velocities[joint] = 0.0

        return velocities

    def draw_poses(self, frame, poses):
        """
        Draw skeleton overlays on the frame.

        Args:
            frame: BGR numpy array
            poses: dict from estimate()

        Returns:
            annotated frame
        """
        for person_id, pose_data in poses.items():
            landmarks = pose_data["landmarks"]

            # Color based on person ID (same hue scheme as detector)
            color_hue = (person_id * 47) % 180
            skeleton_color = cv2.cvtColor(
                np.uint8([[[color_hue, 180, 220]]]),
                cv2.COLOR_HSV2BGR
            )[0][0].tolist()
            joint_color = cv2.cvtColor(
                np.uint8([[[color_hue, 255, 255]]]),
                cv2.COLOR_HSV2BGR
            )[0][0].tolist()

            # Draw skeleton connections
            for start_name, end_name in POSE_CONNECTIONS:
                if start_name in landmarks and end_name in landmarks:
                    pt1 = landmarks[start_name]
                    pt2 = landmarks[end_name]
                    cv2.line(frame, pt1, pt2, skeleton_color, 2, cv2.LINE_AA)

            # Draw joint points
            for name, (px, py) in landmarks.items():
                cv2.circle(frame, (px, py), 4, joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 5, skeleton_color, 1, cv2.LINE_AA)

            # Draw body tilt indicator (small text near the person)
            tilt = pose_data.get("body_tilt", 0)
            bbox = pose_data["bbox"]
            tilt_color = (0, 255, 0) if tilt < 30 else (0, 165, 255) if tilt < 60 else (0, 0, 255)
            cv2.putText(
                frame,
                f"Tilt: {tilt:.0f}",
                (int(bbox[0]), int(bbox[3]) + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                tilt_color,
                1,
            )

        return frame

    def cleanup(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
        print("[PoseEstimator] Closed")
