import requests
import logging
from pathlib import Path
from typing import Dict, Any

class Notifier:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def send_notification(self, detection_data: Dict[str, Any]):
        """Send Pushover notification for pigeon detection"""
        if not self.config.pushover_enabled:
            return

        try:
            self._send_pushover(detection_data)
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")

    def _send_pushover(self, detection_data: Dict[str, Any]):
        """Send Pushover notification"""
        if not self.config.pushover_user_key or not self.config.pushover_api_token:
            self.logger.warning("Pushover credentials not configured")
            return

        try:
            message = f"Pigeon detected with {detection_data['confidence']:.2%} confidence at {detection_data['timestamp']}"

            payload = {
                'token': self.config.pushover_api_token,
                'user': self.config.pushover_user_key,
                'message': message,
                'title': 'Pigeon Detection Alert',
                'priority': 1,
                'sound': 'pushover'
            }

            # Attach image if available
            files = {}
            if detection_data.get('image_path') and Path(detection_data['image_path']).exists():
                files['attachment'] = open(detection_data['image_path'], 'rb')

            response = requests.post(
                'https://api.pushover.net/1/messages.json',
                data=payload,
                files=files
            )

            if files:
                files['attachment'].close()

            response.raise_for_status()
            self.logger.info("Pushover notification sent successfully")

        except Exception as e:
            self.logger.error(f"Pushover notification failed: {e}")