# AI CCTV Anomaly Detection — Phase 1 Walkthrough

## What Was Built

**Phase 1: Project Setup & Video Capture Pipeline** — the foundational layer of the system.

### Files Created

| File | Purpose |
|------|---------|
| [requirements.txt](file:///Users/madhurjya/Documents/Bitcoin2009/requirements.txt) | Python dependencies |
| [config.py](file:///Users/madhurjya/Documents/Bitcoin2009/config.py) | Central configuration (thresholds, ports, Telegram creds) |
| [capture.py](file:///Users/madhurjya/Documents/Bitcoin2009/capture.py) | Video capture with FPS throttling & auto-reconnect |
| [dashboard.py](file:///Users/madhurjya/Documents/Bitcoin2009/dashboard.py) | Flask + SocketIO server (MJPEG stream, WebSocket alerts) |
| [main.py](file:///Users/madhurjya/Documents/Bitcoin2009/main.py) | Entry point wiring the pipeline |
| [templates/index.html](file:///Users/madhurjya/Documents/Bitcoin2009/templates/index.html) | Dashboard UI |
| [static/style.css](file:///Users/madhurjya/Documents/Bitcoin2009/static/style.css) | Dark-themed modern styling |

### How to Run

```bash
cd /Users/madhurjya/Documents/Bitcoin2009
source venv/bin/activate
python main.py
# Open http://localhost:5050
```

## Verification

### Dashboard Screenshot
![Phase 1 Dashboard — Live webcam feed with detection controls and alert log](/Users/madhurjya/.gemini/antigravity/brain/46899fac-4e17-46d2-bf94-e09394516254/cctv_dashboard_verify_1773400945654.png)

### Dashboard Recording
![Phase 1 verification recording](/Users/madhurjya/.gemini/antigravity/brain/46899fac-4e17-46d2-bf94-e09394516254/dashboard_phase1_1773400883858.webp)

### Confirmed Working
- ✅ Live webcam feed streaming at 640×360 @ ~10 FPS
- ✅ Dashboard accessible at `http://localhost:5050`
- ✅ Detection control toggles (Fight, Fall, Theft) rendered
- ✅ Alert log section with empty state
- ✅ SocketIO connection active (green "Live" indicator)
- ✅ Graceful fallback when camera not available

## Next: Phase 2 — Human Detection (YOLOv8)
