"""Notification handler for sending alerts"""
import logging
from pathlib import Path
from app.event import EventBus, DetectionEvent
from app.notifier import Notifier


class NotificationEventHandler:
    """Handles sending notifications for detections"""

    def __init__(self, config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.notifier = Notifier(config)

        # Subscribe to detection events
        self.event_bus.subscribe(DetectionEvent, self._on_detection)

    async def _on_detection(self, event: DetectionEvent):
        """Handle detection events by sending notifications"""
        if not self.config.pushover_enabled:
            return

        try:
            # Construct expected image path based on timestamp
            timestamp_str = event.timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"pigeon_detection_{timestamp_str}_{event.confidence:.3f}.{self.config.image_format}"
            image_path = str(Path(self.config.detection_folder) / filename)

            detection_data = {
                'timestamp': event.timestamp.isoformat(),
                'confidence': event.confidence,
                'image_path': image_path if self.config.save_detections else None
            }

            # Send notification (runs in executor to avoid blocking)
            self.notifier.send_notification(detection_data)

        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
