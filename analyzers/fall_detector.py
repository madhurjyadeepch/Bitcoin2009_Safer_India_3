"""
Fall / Accident detection analyzer.
Detects falls using body tilt, bounding box aspect ratio, and vertical drop speed.
"""

import numpy as np
from collections import defaultdict, deque
import config


class FallDetector:
    """
    Detects falls/accidents using:
    - Body tilt angle exceeding threshold (torso nearly horizontal)
    - Bounding box aspect ratio flip (tall -> wide)
    - Rapid vertical drop of hip position
    - Persistence check: condition must hold for N frames to avoid false positives
    """

    def __init__(self):
        print("[FallDetector] Initialized")
        self.tilt_threshold = config.FALL_BODY_ANGLE_THRESHOLD
        self.aspect_threshold = config.FALL_ASPECT_RATIO_THRESHOLD
        self.drop_threshold = config.FALL_VERTICAL_DROP_THRESHOLD
        self.persistence_frames = config.FALL_PERSISTENCE_FRAMES

        # Per-person history: {person_id: deque of booleans (is_fall_frame)}
        self._fall_history = defaultdict(lambda: deque(maxlen=self.persistence_frames + 5))
        # Per-person hip Y history for vertical drop detection
        self._hip_history = defaultdict(lambda: deque(maxlen=10))
        # Track alerted persons to avoid spamming (person_id -> last alert frame)
        self._alerted = {}
        self._frame_counter = 0

    def analyze(self, tracked_persons, poses):
        """
        Analyze current frame for fall activity.

        Args:
            tracked_persons: dict {id: {"bbox": (x1,y1,x2,y2), "centroid": (cx,cy)}}
            poses: dict {id: {"landmarks": ..., "angles": ..., "body_tilt": float, ...}}

        Returns:
            list of fall events: [{
                "type": "fall",
                "confidence": float,
                "person": int,
                "bbox": (x1, y1, x2, y2),
            }]
        """
        self._frame_counter += 1
        events = []
        active_ids = set()

        for person_id, data in tracked_persons.items():
            active_ids.add(person_id)
            bbox = data["bbox"]
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1

            if bh < 1:
                continue

            # ── Signal 1: Bounding box aspect ratio ──
            aspect_ratio = bw / bh
            is_wide = aspect_ratio > self.aspect_threshold

            # ── Signal 2: Body tilt from pose ──
            body_tilt = 0.0
            if person_id in poses:
                body_tilt = poses[person_id].get("body_tilt", 0.0)
            is_tilted = body_tilt > self.tilt_threshold

            # ── Signal 3: Rapid vertical drop (hip Y) ──
            has_vertical_drop = False
            drop_speed = 0.0
            if person_id in poses:
                landmarks = poses[person_id].get("landmarks", {})
                if "left_hip" in landmarks and "right_hip" in landmarks:
                    mid_hip_y = (landmarks["left_hip"][1] + landmarks["right_hip"][1]) / 2
                    self._hip_history[person_id].append(mid_hip_y)

                    if len(self._hip_history[person_id]) >= 3:
                        prev_y = self._hip_history[person_id][-3]
                        curr_y = mid_hip_y
                        drop_speed = (curr_y - prev_y) / 2.0  # positive = downward
                        if drop_speed > self.drop_threshold:
                            has_vertical_drop = True

            # ── Combine signals ──
            # Fall = at least 2 of 3 signals active
            signal_count = sum([is_wide, is_tilted, has_vertical_drop])
            is_fall_frame = signal_count >= 2

            self._fall_history[person_id].append(is_fall_frame)

            # ── Persistence check ──
            history = self._fall_history[person_id]
            if len(history) < self.persistence_frames:
                continue

            recent = list(history)[-self.persistence_frames:]
            fall_ratio = sum(recent) / len(recent)

            # Need most of the recent frames to show fall signals
            if fall_ratio < 0.6:
                continue

            # Check cooldown: don't re-alert same person within 60 frames
            if person_id in self._alerted:
                if self._frame_counter - self._alerted[person_id] < 60:
                    continue

            # Calculate confidence
            tilt_score = min(body_tilt / 90.0, 1.0) if is_tilted else 0
            aspect_score = min(aspect_ratio / 2.0, 1.0) if is_wide else 0
            drop_score = min(drop_speed / (self.drop_threshold * 2), 1.0) if has_vertical_drop else 0

            confidence = (
                0.35 * tilt_score +
                0.25 * aspect_score +
                0.2 * drop_score +
                0.2 * fall_ratio
            )

            if confidence > 0.4:
                self._alerted[person_id] = self._frame_counter
                events.append({
                    "type": "fall",
                    "confidence": round(float(confidence), 3),
                    "person": person_id,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

        # Cleanup stale
        stale = set(self._fall_history.keys()) - active_ids
        for sid in stale:
            del self._fall_history[sid]
            if sid in self._hip_history:
                del self._hip_history[sid]
            if sid in self._alerted:
                del self._alerted[sid]

        return events
