"""
AI CCTV Anomaly Detection System — Main Entry Point

Phase 1: Video capture + live dashboard display.
Phase 2: YOLOv8 person detection with centroid tracking.
Subsequent phases will add pose estimation and anomaly detectors.
"""

import cv2
import signal
import sys
import time
from capture import VideoCapture
from detector import PersonDetector
import dashboard
import config


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n[Main] Shutting down...")
    sys.exit(0)


def draw_info_overlay(frame, fps=0):
    """Draw status info on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Status text
    cv2.putText(frame, f"AI CCTV Monitor | {w}x{h} | {fps:.0f} FPS",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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

    # Initialize person detector
    person_detector = PersonDetector()

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

                # ── Phase 2: Person Detection ──
                tracked_persons, raw_detections = person_detector.detect(frame)
                frame = person_detector.draw_detections(frame, tracked_persons)

                # ── Future phases will plug in here ──
                # Phase 3: poses = pose_estimator.estimate(frame, tracked_persons)
                # Phase 4-6: anomalies = analyze(poses, tracked_persons)
                # Phase 7: alert_manager.process(anomalies, frame)

                # Draw overlay
                frame = draw_info_overlay(frame, current_fps)

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
        print("[Main] Stopped.")


if __name__ == "__main__":
    main()
