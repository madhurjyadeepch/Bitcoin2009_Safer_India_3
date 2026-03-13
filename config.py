"""
Central configuration for the AI CCTV Anomaly Detection System.
All thresholds, model paths, and settings are managed here.
"""

# ──────────────────────────────────────────────
# Video Source
# ──────────────────────────────────────────────
# 0 = default webcam, or provide a path to a video file / RTSP URL
VIDEO_SOURCE = 0

# Target FPS for processing (lower = less CPU, higher = smoother)
PROCESS_FPS = 15

# Frame resize width for processing (keeps aspect ratio)
FRAME_WIDTH = 640

# ──────────────────────────────────────────────
# YOLOv8 Detection
# ──────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"  # nano model — fast on M2
YOLO_CONFIDENCE = 0.5       # minimum confidence for person detection
YOLO_PERSON_CLASS = 0       # COCO class ID for "person"

# ──────────────────────────────────────────────
# Pose Estimation (MediaPipe)
# ──────────────────────────────────────────────
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# ──────────────────────────────────────────────
# Fight Detection
# ──────────────────────────────────────────────
FIGHT_PROXIMITY_THRESHOLD = 150    # pixels — max distance between two persons to consider interaction
FIGHT_WRIST_VELOCITY_THRESHOLD = 30  # pixels/frame — rapid arm motion threshold
FIGHT_FRAME_WINDOW = 15            # number of frames to analyze for temporal smoothing
FIGHT_CONFIDENCE_THRESHOLD = 0.6   # minimum confidence to trigger fight alert

# ──────────────────────────────────────────────
# Fall / Accident Detection
# ──────────────────────────────────────────────
FALL_ASPECT_RATIO_THRESHOLD = 1.0  # bbox width/height > this = likely fallen
FALL_VERTICAL_DROP_THRESHOLD = 40  # pixels/frame — rapid vertical drop
FALL_BODY_ANGLE_THRESHOLD = 60     # degrees — torso angle from vertical
FALL_PERSISTENCE_FRAMES = 8        # must persist for this many frames (~0.5s at 15fps)

# ──────────────────────────────────────────────
# Theft Detection (Grab & Run)
# ──────────────────────────────────────────────
THEFT_SPEED_SPIKE_THRESHOLD = 35   # pixels/frame — sudden speed increase
THEFT_PROXIMITY_THRESHOLD = 120    # pixels — was near another person/object before running
THEFT_FRAME_WINDOW = 20            # frames to analyze for speed pattern

# ──────────────────────────────────────────────
# Alert System
# ──────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 30        # prevent duplicate alerts within this window
ALERTS_DIR = "alerts"              # directory to save event screenshots

# ──────────────────────────────────────────────
# Telegram Bot
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = ""            # Your Telegram bot token from @BotFather
TELEGRAM_CHAT_ID = ""              # Chat ID or group ID to send alerts to

# ──────────────────────────────────────────────
# Web Dashboard
# ──────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5050

# ──────────────────────────────────────────────
# Tracking
# ──────────────────────────────────────────────
TRACKER_MAX_DISAPPEARED = 30       # frames before a tracked person is removed
TRACKER_MAX_DISTANCE = 80          # max pixels to match centroids across frames
