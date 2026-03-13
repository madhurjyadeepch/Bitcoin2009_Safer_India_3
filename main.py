"""
AI CCTV Anomaly Detection System — Main Entry Point

Phase 1: Video capture + live dashboard display.
Phase 2: YOLOv8 person detection with centroid tracking.
Phase 3: MediaPipe pose estimation with skeleton overlay.
Phase 4: Fight detection using pose + motion signals.
Subsequent phases will add fall/accident and theft detectors.
"""

import cv2
import signal
import sys
import time
from datetime import datetime
from capture import VideoCapture
from detector import PersonDetector
from pose_estimator import PoseEstimator
from analyzers.fight_detector import FightDetector
import dashboard
import config


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n[Main] Shutting down...")
    sys.exit(0)


def draw_info_overlay(frame, fps=0, alert_text=None):
    """Draw status info on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Status text
    cv2.putText(frame, f"AI CCTV Monitor | {w}x{h} | {fps:.0f} FPS",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Alert banner if active
    if alert_text:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 40), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, alert_text, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def draw_fight_overlay(frame, events):
    """Draw fight detection overlays — bounding box and label."""
    for event in events:
        x1, y1, x2, y2 = [int(v) for v in event["bbox"]]
        conf = event["confidence"]

        # Red bounding box around the fight area
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Label
        label = f"FIGHT {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 12, y1), (0, 0, 200), -1)
        cv2.putText(frame, label, (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def main():
    """Main pipeline loop."""
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 50)
    print("  AI CCTV Anomaly Detection System")
    print("=" * 50)
    print()

    # Start the web dashboard in a background thread
    dashboard.start_dashboard(threaded=True)
    print()

    # Initialize detection pipeline
    person_detector = PersonDetector()
    pose_estimator = PoseEstimator()
    fight_detector = FightDetector()

    # Start video capture
    cap = VideoCapture()
    camera_available = True
    try:
        cap.start()
    except RuntimeError as e:
        print(f"[Main] Warning: {e}")
        print("[Main] Dashboard will run without live feed.")
        print("[Main] To fix: grant camera access in System Settings > Privacy > Camera")
        print("[Main] Or set VIDEO_SOURCE in config.py to a video file path.\n")
        camera_available = False

        # Send a placeholder frame so the dashboard shows something
        import numpy as np
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Camera Available", (140, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
        cv2.putText(placeholder, "Grant camera access or set VIDEO_SOURCE in config.py", (50, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        dashboard.update_frame(placeholder)

    print(f"[Main] Processing at {config.PROCESS_FPS} FPS")
    print(f"[Main] Open http://localhost:{config.DASHBOARD_PORT} in your browser")
    print("[Main] Press Ctrl+C to stop\n")

    # FPS tracking
    frame_count = 0
    fps_start = time.time()
    current_fps = 0
    active_alert_text = None
    alert_clear_time = 0

    try:
        if not camera_available:
            # Keep the process alive so the dashboard stays up
            print("[Main] Running dashboard-only mode. Waiting for Ctrl+C...")
            while True:
                time.sleep(1)
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Main] No more frames. Stopping.")
                    break

                # Get current toggle states
                toggles = dashboard.get_toggles()

                # ── Phase 2: Person Detection ──
                tracked_persons, raw_detections = person_detector.detect(frame)
                frame = person_detector.draw_detections(frame, tracked_persons)

                # ── Phase 3: Pose Estimation ──
                poses = pose_estimator.estimate(frame, tracked_persons)
                frame = pose_estimator.draw_poses(frame, poses)

                # ── Phase 4: Fight Detection ──
                fight_events = []
                if toggles.get("fight", True):
                    fight_events = fight_detector.analyze(tracked_persons, poses)
                    if fight_events:
                        frame = draw_fight_overlay(frame, fight_events)
                        # Send alerts to dashboard
                        for event in fight_events:
                            alert_data = {
                                "type": "fight",
                                "confidence": event["confidence"],
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "persons": event["persons"],
                            }
                            dashboard.emit_alert(alert_data)
                            active_alert_text = f"⚠ FIGHT DETECTED — Confidence: {event['confidence']*100:.0f}%"
                            alert_clear_time = time.time() + 3  # show for 3 seconds

                # ── Future phases will plug in here ──
                # Phase 5: fall_events = fall_detector.analyze(tracked_persons, poses)
                # Phase 6: theft_events = theft_detector.analyze(tracked_persons, poses)
                # Phase 7: alert_manager.process(all_events, frame)

                # Clear alert text after timeout
                if active_alert_text and time.time() > alert_clear_time:
                    active_alert_text = None

                # Draw overlay
                frame = draw_info_overlay(frame, current_fps, active_alert_text)

                # Send frame to dashboard
                dashboard.update_frame(frame)

                # FPS calculation
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    current_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        if camera_available:
            cap.release()
        pose_estimator.cleanup()
        print("[Main] Stopped.")


if __name__ == "__main__":
    main()
