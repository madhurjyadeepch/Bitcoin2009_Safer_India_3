"""
Flask + SocketIO web dashboard for live video feed and alert log.
"""

import cv2
import threading
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import config

app = Flask(__name__)
app.config["SECRET_KEY"] = "cctv-anomaly-detection"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared state
_frame_lock = threading.Lock()
_current_frame = None
_detection_toggles = {
    "fight": True,
    "fall": True,
    "theft": True,
}


def update_frame(frame):
    """Called by the main pipeline to update the frame displayed on dashboard."""
    global _current_frame
    with _frame_lock:
        _current_frame = frame.copy()


def get_frame():
    """Get the latest frame."""
    with _frame_lock:
        if _current_frame is None:
            return None
        return _current_frame.copy()


def emit_alert(alert_data):
    """Push an alert event to connected dashboard clients."""
    socketio.emit("new_alert", alert_data)


def generate_mjpeg():
    """Generator that yields MJPEG frames for the video feed endpoint."""
    while True:
        frame = get_frame()
        if frame is None:
            # Send a blank frame if no video yet
            import numpy as np
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for video...", (150, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        # Small sleep to control stream rate
        import time
        time.sleep(1.0 / config.PROCESS_FPS)


@app.route("/")
def index():
    """Serve the dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@socketio.on("toggle_detection")
def handle_toggle(data):
    """Handle detection toggle from the dashboard."""
    detection_type = data.get("type")
    enabled = data.get("enabled", True)
    if detection_type in _detection_toggles:
        _detection_toggles[detection_type] = enabled
        print(f"[Dashboard] {detection_type} detection {'enabled' if enabled else 'disabled'}")
        socketio.emit("toggle_update", {"type": detection_type, "enabled": enabled})


def get_toggles():
    """Return current detection toggle states."""
    return _detection_toggles.copy()


def start_dashboard(threaded=True):
    """Start the Flask dashboard server."""
    if threaded:
        thread = threading.Thread(
            target=lambda: socketio.run(
                app,
                host=config.DASHBOARD_HOST,
                port=config.DASHBOARD_PORT,
                allow_unsafe_werkzeug=True,
                use_reloader=False
            ),
            daemon=True
        )
        thread.start()
        print(f"[Dashboard] Running at http://localhost:{config.DASHBOARD_PORT}")
        return thread
    else:
        socketio.run(
            app,
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            allow_unsafe_werkzeug=True,
            use_reloader=False
        )
