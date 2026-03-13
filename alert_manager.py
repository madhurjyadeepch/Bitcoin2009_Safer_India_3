"""
Alert manager — de-duplicates alerts, saves screenshots, and sends Telegram messages.
"""

import os
import cv2
import time
import threading
from datetime import datetime
import config

# Try importing telegram bot (optional — works without it)
try:
    import telegram
    import asyncio
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class AlertManager:
    """
    Manages alert lifecycle:
    - De-duplicates alerts using cooldown periods
    - Saves annotated screenshots to alerts/ directory
    - Sends Telegram messages with screenshots (if configured)
    - Pushes events to dashboard via callback
    """

    def __init__(self, dashboard_callback=None):
        print("[AlertManager] Initializing...")
        self.cooldown = config.ALERT_COOLDOWN_SECONDS
        self.alerts_dir = config.ALERTS_DIR
        self.dashboard_callback = dashboard_callback

        # Create alerts directory
        os.makedirs(self.alerts_dir, exist_ok=True)

        # Cooldown tracking: {"fight": last_alert_time, ...}
        self._last_alert_time = {}

        # Telegram bot setup
        self._telegram_bot = None
        self._telegram_chat_id = config.TELEGRAM_CHAT_ID
        if TELEGRAM_AVAILABLE and config.TELEGRAM_BOT_TOKEN:
            try:
                self._telegram_bot = telegram.Bot(token=config.TELEGRAM_BOT_TOKEN)
                print("[AlertManager] Telegram bot configured")
            except Exception as e:
                print(f"[AlertManager] Telegram bot init failed: {e}")
        else:
            if not TELEGRAM_AVAILABLE:
                print("[AlertManager] python-telegram-bot not installed — Telegram alerts disabled")
            elif not config.TELEGRAM_BOT_TOKEN:
                print("[AlertManager] No TELEGRAM_BOT_TOKEN set — Telegram alerts disabled")

        # Stats
        self._total_alerts = 0
        print("[AlertManager] Ready")

    def process(self, events, frame):
        """
        Process a list of anomaly events from all detectors.

        Args:
            events: list of event dicts [{"type": str, "confidence": float, ...}]
            frame: current BGR frame for screenshot
        """
        for event in events:
            event_type = event.get("type", "unknown")

            # Check cooldown
            if not self._check_cooldown(event_type):
                continue

            self._total_alerts += 1
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%H:%M:%S")
            timestamp_file = timestamp.strftime("%Y%m%d_%H%M%S")

            # Save screenshot
            screenshot_path = self._save_screenshot(frame, event, timestamp_file)

            # Build alert data
            alert_data = {
                "type": event_type,
                "confidence": event.get("confidence", 0),
                "timestamp": timestamp_str,
                "screenshot": screenshot_path,
            }

            # Push to dashboard
            if self.dashboard_callback:
                self.dashboard_callback(alert_data)

            # Send Telegram (async, non-blocking)
            if self._telegram_bot and self._telegram_chat_id:
                self._send_telegram_async(alert_data, screenshot_path)

            # Log to console
            print(f"[AlertManager] ⚠ {event_type.upper()} detected "
                  f"(conf: {event.get('confidence', 0):.0%}) — "
                  f"screenshot saved: {screenshot_path}")

    def _check_cooldown(self, event_type):
        """Check if an event type is within cooldown period."""
        now = time.time()
        last = self._last_alert_time.get(event_type, 0)
        if now - last < self.cooldown:
            return False
        self._last_alert_time[event_type] = now
        return True

    def _save_screenshot(self, frame, event, timestamp):
        """Save an annotated screenshot for the alert."""
        event_type = event.get("type", "unknown")
        filename = f"{event_type}_{timestamp}.jpg"
        filepath = os.path.join(self.alerts_dir, filename)

        # Draw alert info on screenshot
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Red banner at bottom
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        label = f"ALERT: {event_type.upper()} | Conf: {event.get('confidence', 0):.0%} | {timestamp}"
        cv2.putText(annotated, label, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Draw bbox if present
        if "bbox" in event:
            bx1, by1, bx2, by2 = [int(v) for v in event["bbox"]]
            color = {
                "fight": (0, 0, 255),
                "fall": (0, 165, 255),
                "theft": (180, 0, 220),
            }.get(event_type, (0, 0, 255))
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 3)

        cv2.imwrite(filepath, annotated)
        return filepath

    def _send_telegram_async(self, alert_data, screenshot_path):
        """Send Telegram message in a background thread."""
        thread = threading.Thread(
            target=self._send_telegram,
            args=(alert_data, screenshot_path),
            daemon=True
        )
        thread.start()

    def _send_telegram(self, alert_data, screenshot_path):
        """Send Telegram message with screenshot."""
        try:
            event_type = alert_data["type"]
            confidence = alert_data.get("confidence", 0)
            timestamp = alert_data.get("timestamp", "")

            icons = {"fight": "🥊", "fall": "⚠️", "theft": "🏃"}
            icon = icons.get(event_type, "🔴")

            message = (
                f"{icon} *{event_type.upper()} DETECTED*\n\n"
                f"📊 Confidence: {confidence:.0%}\n"
                f"🕐 Time: {timestamp}\n"
                f"📍 Camera: Active feed\n\n"
                f"_AI CCTV Anomaly Detection System_"
            )

            # Run async bot methods in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Send photo with caption
                if os.path.exists(screenshot_path):
                    with open(screenshot_path, "rb") as photo:
                        loop.run_until_complete(
                            self._telegram_bot.send_photo(
                                chat_id=self._telegram_chat_id,
                                photo=photo,
                                caption=message,
                                parse_mode="Markdown"
                            )
                        )
                else:
                    loop.run_until_complete(
                        self._telegram_bot.send_message(
                            chat_id=self._telegram_chat_id,
                            text=message,
                            parse_mode="Markdown"
                        )
                    )
            finally:
                loop.close()

            print(f"[AlertManager] Telegram alert sent for {event_type}")
        except Exception as e:
            print(f"[AlertManager] Telegram send failed: {e}")

    @property
    def total_alerts(self):
        return self._total_alerts
