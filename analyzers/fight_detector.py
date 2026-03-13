"""
Fight detection analyzer.
Detects fights using proximity, rapid arm motion, and aggressive pose signals.
"""

import numpy as np
from collections import defaultdict, deque
import config


class FightDetector:
    """
    Detects fights between persons using:
    - Proximity check: two persons within close distance
    - Rapid arm motion: wrist velocity exceeds threshold
    - Aggressive pose: arm extension angles consistent with punching/pushing
    - Temporal smoothing over a sliding window of frames
    """

    def __init__(self):
        print("[FightDetector] Initialized")
        self.proximity_threshold = config.FIGHT_PROXIMITY_THRESHOLD
        self.velocity_threshold = config.FIGHT_WRIST_VELOCITY_THRESHOLD
        self.confidence_threshold = config.FIGHT_CONFIDENCE_THRESHOLD
        self.frame_window = config.FIGHT_FRAME_WINDOW

        # History of fight signals for temporal smoothing
        # Key: frozenset({id1, id2}), Value: deque of booleans
        self._pair_history = defaultdict(lambda: deque(maxlen=self.frame_window))

    def analyze(self, tracked_persons, poses):
        """
        Analyze current frame for fight activity.

        Args:
            tracked_persons: dict {id: {"bbox": (x1,y1,x2,y2), "centroid": (cx,cy)}}
            poses: dict {id: {"landmarks": ..., "angles": ..., "velocities": ..., "bbox": ...}}

        Returns:
            list of fight events: [
                {
                    "type": "fight",
                    "confidence": float,
                    "persons": [id1, id2],
                    "bbox": (x1, y1, x2, y2),  # bounding box enclosing both persons
                }
            ]
        """
        events = []
        person_ids = list(tracked_persons.keys())
        active_pairs = set()

        # Check all pairs of detected persons
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                id1, id2 = person_ids[i], person_ids[j]
                pair_key = frozenset({id1, id2})
                active_pairs.add(pair_key)

                # Get centroids
                c1 = tracked_persons[id1]["centroid"]
                c2 = tracked_persons[id2]["centroid"]

                # ── Signal 1: Proximity ──
                distance = np.sqrt(
                    (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2
                )
                is_close = distance < self.proximity_threshold

                # ── Signal 2: Rapid arm motion ──
                has_rapid_motion = False
                rapid_motion_score = 0.0

                for pid in (id1, id2):
                    if pid in poses:
                        vel = poses[pid].get("velocities", {})
                        lw = vel.get("left_wrist", 0)
                        rw = vel.get("right_wrist", 0)
                        max_wrist_vel = max(lw, rw)
                        if max_wrist_vel > self.velocity_threshold:
                            has_rapid_motion = True
                            rapid_motion_score = max(
                                rapid_motion_score,
                                min(max_wrist_vel / (self.velocity_threshold * 2), 1.0),
                            )

                # ── Signal 3: Aggressive pose (extended arms) ──
                has_aggressive_pose = False
                aggressive_score = 0.0

                for pid in (id1, id2):
                    if pid in poses:
                        angles = poses[pid].get("angles", {})
                        # Extended arms: elbow angle > 140° (nearly straight)
                        # or shoulder angle > 90° (arm raised high)
                        for side in ("left", "right"):
                            elbow_angle = angles.get(f"{side}_elbow", 180)
                            shoulder_angle = angles.get(f"{side}_shoulder", 0)
                            if elbow_angle > 140 and shoulder_angle > 70:
                                has_aggressive_pose = True
                                aggressive_score = max(
                                    aggressive_score,
                                    min(shoulder_angle / 120.0, 1.0),
                                )

                # ── Combine signals ──
                # Fight requires at least proximity + one other signal
                is_fight_frame = is_close and (has_rapid_motion or has_aggressive_pose)
                self._pair_history[pair_key].append(is_fight_frame)

                # ── Temporal smoothing ──
                history = self._pair_history[pair_key]
                if len(history) < 3:
                    continue

                fight_ratio = sum(history) / len(history)

                # Calculate composite confidence
                proximity_score = max(0, 1.0 - (distance / self.proximity_threshold)) if is_close else 0
                confidence = (
                    0.3 * proximity_score +
                    0.35 * rapid_motion_score +
                    0.15 * aggressive_score +
                    0.2 * fight_ratio
                )

                if confidence >= self.confidence_threshold and fight_ratio > 0.3:
                    # Compute enclosing bounding box
                    b1 = tracked_persons[id1]["bbox"]
                    b2 = tracked_persons[id2]["bbox"]
                    enc_bbox = (
                        min(b1[0], b2[0]),
                        min(b1[1], b2[1]),
                        max(b1[2], b2[2]),
                        max(b1[3], b2[3]),
                    )

                    events.append({
                        "type": "fight",
                        "confidence": round(float(confidence), 3),
                        "persons": sorted([id1, id2]),
                        "bbox": enc_bbox,
                    })

        # Clean up stale pair history
        stale = set(self._pair_history.keys()) - active_pairs
        for key in stale:
            del self._pair_history[key]

        return events
