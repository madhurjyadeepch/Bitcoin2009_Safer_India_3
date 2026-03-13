"""
Theft detection analyzer (grab & run).
Detects suspicious grab-and-run patterns using trajectory and pose signals.
"""

import numpy as np
from collections import defaultdict, deque
import config


class TheftDetector:
    """
    Detects grab-and-run theft using:
    - Person was near another person (proximity)
    - Sudden speed increase (running away)
    - Rapid arm extension (grab gesture)
    - Combines trajectory + pose signals
    """

    def __init__(self):
        print("[TheftDetector] Initialized")
        self.speed_threshold = config.THEFT_SPEED_SPIKE_THRESHOLD
        self.proximity_threshold = config.THEFT_PROXIMITY_THRESHOLD
        self.frame_window = config.THEFT_FRAME_WINDOW

        # Per-person speed history: {person_id: deque of speeds}
        self._speed_history = defaultdict(lambda: deque(maxlen=self.frame_window))
        # Per-person centroid history for speed calculation
        self._centroid_history = defaultdict(lambda: deque(maxlen=self.frame_window))
        # Was-near tracker: {person_id: set of person_ids they were recently near}
        self._proximity_history = defaultdict(lambda: deque(maxlen=self.frame_window))
        # Cooldown
        self._alerted = {}
        self._frame_counter = 0

    def analyze(self, tracked_persons, poses):
        """
        Analyze current frame for theft patterns.

        Args:
            tracked_persons: dict {id: {"bbox": (x1,y1,x2,y2), "centroid": (cx,cy)}}
            poses: dict {id: {"landmarks": ..., "velocities": ..., ...}}

        Returns:
            list of theft events: [{
                "type": "theft",
                "confidence": float,
                "person": int,
                "bbox": (x1, y1, x2, y2),
            }]
        """
        self._frame_counter += 1
        events = []
        active_ids = set()
        person_ids = list(tracked_persons.keys())

        # Pre-compute: who is near whom right now
        near_pairs = set()
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                id1, id2 = person_ids[i], person_ids[j]
                c1 = tracked_persons[id1]["centroid"]
                c2 = tracked_persons[id2]["centroid"]
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                if dist < self.proximity_threshold:
                    near_pairs.add((id1, id2))
                    near_pairs.add((id2, id1))

        for person_id, data in tracked_persons.items():
            active_ids.add(person_id)
            centroid = data["centroid"]
            bbox = data["bbox"]

            # Track centroid history
            self._centroid_history[person_id].append(centroid)

            # Record who is near this person right now
            near_now = {pid for (pid, other) in near_pairs if other == person_id}
            # Actually: we want IDs where (person_id, other) is in near_pairs
            near_now = set()
            for (a, b) in near_pairs:
                if a == person_id:
                    near_now.add(b)
            self._proximity_history[person_id].append(len(near_now) > 0)

            # Calculate speed (pixels/frame)
            speed = 0.0
            ch = self._centroid_history[person_id]
            if len(ch) >= 2:
                prev = ch[-2]
                curr = ch[-1]
                speed = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            self._speed_history[person_id].append(speed)

            # ── Signal 1: Was near someone recently, now moving away fast ──
            was_near_recently = False
            prox_hist = self._proximity_history[person_id]
            if len(prox_hist) >= 5:
                # Was near someone in the earlier part of the window
                early = list(prox_hist)[:len(prox_hist)//2]
                was_near_recently = any(early)

            # ── Signal 2: Sudden speed spike ──
            has_speed_spike = False
            speed_score = 0.0
            sh = self._speed_history[person_id]
            if len(sh) >= 5:
                recent_speeds = list(sh)[-3:]
                earlier_speeds = list(sh)[:-3]
                avg_recent = np.mean(recent_speeds)
                avg_earlier = np.mean(earlier_speeds) if earlier_speeds else 0

                # Speed spike = current speed is much higher than previous average
                if avg_recent > self.speed_threshold and avg_recent > avg_earlier * 2.5:
                    has_speed_spike = True
                    speed_score = min(avg_recent / (self.speed_threshold * 2), 1.0)

            # ── Signal 3: Rapid arm motion (grab gesture) ──
            has_grab_motion = False
            grab_score = 0.0
            if person_id in poses:
                vel = poses[person_id].get("velocities", {})
                max_wrist = max(vel.get("left_wrist", 0), vel.get("right_wrist", 0))
                if max_wrist > self.speed_threshold * 0.8:
                    has_grab_motion = True
                    grab_score = min(max_wrist / (self.speed_threshold * 1.5), 1.0)

            # ── Combine: need proximity history + speed spike ──
            if not (was_near_recently and has_speed_spike):
                continue

            # Cooldown check
            if person_id in self._alerted:
                if self._frame_counter - self._alerted[person_id] < 90:
                    continue

            confidence = (
                0.35 * speed_score +
                0.35 * (1.0 if was_near_recently else 0.0) +
                0.3 * grab_score
            )

            if confidence > 0.45:
                self._alerted[person_id] = self._frame_counter
                events.append({
                    "type": "theft",
                    "confidence": round(float(confidence), 3),
                    "person": person_id,
                    "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                })

        # Cleanup stale
        stale = set(self._speed_history.keys()) - active_ids
        for sid in stale:
            del self._speed_history[sid]
            if sid in self._centroid_history:
                del self._centroid_history[sid]
            if sid in self._proximity_history:
                del self._proximity_history[sid]
            if sid in self._alerted:
                del self._alerted[sid]

        return events
