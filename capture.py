"""
Video capture module.
Reads frames from webcam, video file, or RTSP stream.
Handles reconnection, frame-rate throttling, and camera enumeration.
"""

import cv2
import time
import threading
import config


def enumerate_cameras(max_check=10):
    """
    Detect available camera devices.
    Tries indices 0..max_check and returns list of working camera indices
    with their resolution info.

    Returns:
        list of dicts: [{"index": 0, "name": "Camera 0", "resolution": "1280x720"}, ...]
    """
    cameras = []
    for idx in range(max_check):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Try reading a frame to confirm it actually works
            ret, _ = cap.read()
            if ret:
                cameras.append({
                    "index": idx,
                    "name": f"Camera {idx}",
                    "resolution": f"{w}x{h}",
                })
            cap.release()
        else:
            cap.release()
    return cameras


class VideoCapture:
    """Manages video capture from various sources with frame-rate control."""

    def __init__(self, source=None):
        self.source = source if source is not None else config.VIDEO_SOURCE
        self.cap = None
        self.frame_delay = 1.0 / config.PROCESS_FPS
        self._last_frame_time = 0
        self._lock = threading.Lock()

    def start(self):
        """Open the video source."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        # Set resolution if webcam
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(f"[Capture] Opened video source: {self.source}")
        print(f"[Capture] Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return self

    def switch_source(self, new_source):
        """
        Switch to a different video source (camera index or file path).
        Thread-safe — can be called from the dashboard.
        """
        with self._lock:
            print(f"[Capture] Switching video source to: {new_source}")
            if self.cap is not None:
                self.cap.release()

            self.source = new_source
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                print(f"[Capture] Failed to open source: {new_source}")
                return False

            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Capture] Switched OK — Resolution: {w}x{h}")
            return True

    def read(self):
        """
        Read the next frame, respecting the target FPS.
        Returns (success, frame) tuple.
        Frame is resized to FRAME_WIDTH while keeping aspect ratio.
        """
        with self._lock:
            if self.cap is None or not self.cap.isOpened():
                return False, None

            # Frame-rate throttling
            elapsed = time.time() - self._last_frame_time
            if elapsed < self.frame_delay:
                time.sleep(self.frame_delay - elapsed)

            ret, frame = self.cap.read()
            self._last_frame_time = time.time()

            if not ret:
                # If reading from a file, it might have ended
                if not isinstance(self.source, int):
                    return False, None
                # For live sources, attempt reconnection
                return self._reconnect()

            # Resize for processing
            frame = self._resize(frame)
            return True, frame

    def _resize(self, frame):
        """Resize frame to target width, maintaining aspect ratio."""
        h, w = frame.shape[:2]
        target_w = config.FRAME_WIDTH
        if w == target_w:
            return frame
        scale = target_w / w
        target_h = int(h * scale)
        return cv2.resize(frame, (target_w, target_h))

    def _reconnect(self, max_attempts=5, wait_seconds=2):
        """Attempt to reconnect to the video source."""
        print("[Capture] Connection lost. Attempting reconnection...")
        for attempt in range(1, max_attempts + 1):
            time.sleep(wait_seconds)
            self.cap.release()
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    print(f"[Capture] Reconnected on attempt {attempt}")
                    return True, self._resize(frame)
            print(f"[Capture] Reconnect attempt {attempt}/{max_attempts} failed")
        print("[Capture] Could not reconnect. Giving up.")
        return False, None

    def release(self):
        """Release the video source."""
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                print("[Capture] Released video source")

    def is_opened(self):
        """Check if the video source is open."""
        return self.cap is not None and self.cap.isOpened()

    @property
    def fps(self):
        """Get the source FPS (may differ from processing FPS)."""
        if self.cap is not None:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
