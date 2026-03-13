"""
AI CCTV Anomaly Detection System — Main Entry Point

Full pipeline:
  Phase 1: Video capture + live dashboard display
  Phase 2: YOLOv8 person detection with centroid tracking
  Phase 3: MediaPipe pose estimation with skeleton overlay
  Phase 4: Fight detection using pose + motion signals
  Phase 5: Fall / accident detection
  Phase 6: Theft detection (grab & run)
  Phase 7: Alert system (Telegram + screenshots)

Camera selection available via the dashboard UI.
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
from analyzers.fall_detector import FallDetector
from analyzers.theft_detector import TheftDetector
from alert_manager import AlertManager
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


def draw_anomaly_overlay(frame, events):
    """Draw detection overlays for any anomaly type."""
    colors = {
        "fight": (0, 0, 255),     # Red
        "fall": (0, 165, 255),    # Orange
        "theft": (180, 0, 220),   # Purple
    }
    labels = {
        "fight": "FIGHT",
        "fall": "FALL",
        "theft": "THEFT",
    }

    for event in events:
        if "bbox" not in event:
            continue

        event_type = event.get("type", "unknown")
        x1, y1, x2, y2 = [int(v) for v in event["bbox"]]
        conf = event.get("confidence", 0)
        color = colors.get(event_type, (0, 0, 255))
        label_text = labels.get(event_type, event_type.upper())

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Draw label
        label = f"{label_text} {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 12, y1), color, -1)
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
    fall_detector = FallDetector()
    theft_detector = TheftDetector()
    alert_manager = AlertManager(dashboard_callback=dashboard.emit_alert)

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

        import numpy as np
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Camera Available", (140, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
        cv2.putText(placeholder, "Grant camera access or set VIDEO_SOURCE in config.py", (50, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        dashboard.update_frame(placeholder)

    # Register camera switch callback for the dashboard
    def handle_camera_switch(new_source):
        nonlocal camera_available
        success = cap.switch_source(new_source)
        if success:
            camera_available = True
        return success

    dashboard.set_camera_switch_callback(handle_camera_switch)

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
            print("[Main] Running dashboard-only mode. Waiting for Ctrl+C...")
            while True:
                time.sleep(1)
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if not camera_available:
                        time.sleep(0.1)
                        continue
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

                # ── Phase 4-6: Anomaly Detection ──
                all_events = []

                # Fight Detection
                if toggles.get("fight", True):
                    fight_events = fight_detector.analyze(tracked_persons, poses)
                    all_events.extend(fight_events)

                # Fall Detection
                if toggles.get("fall", True):
                    fall_events = fall_detector.analyze(tracked_persons, poses)
                    all_events.extend(fall_events)

                # Theft Detection
                if toggles.get("theft", True):
                    theft_events = theft_detector.analyze(tracked_persons, poses)
                    all_events.extend(theft_events)

                # Draw anomaly overlays
                if all_events:
                    frame = draw_anomaly_overlay(frame, all_events)

                    # Update alert banner with the highest-confidence event
                    top_event = max(all_events, key=lambda e: e.get("confidence", 0))
                    icons = {"fight": "🥊", "fall": "⚠", "theft": "🏃"}
                    icon = icons.get(top_event["type"], "⚠")
                    active_alert_text = (
                        f"{icon} {top_event['type'].upper()} DETECTED — "
                        f"Confidence: {top_event['confidence']*100:.0f}%"
                    )
                    alert_clear_time = time.time() + 3

                # ── Phase 7: Alert Manager ──
                if all_events:
                    alert_manager.process(all_events, frame)

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
        print(f"[Main] Total alerts sent: {alert_manager.total_alerts}")
        print("[Main] Stopped.")


if __name__ == "__main__":
    main()
