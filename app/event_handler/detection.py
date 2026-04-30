"""Detection handler for saving detection images"""
import cv2
import logging
from pathlib import Path
from app.event import EventBus, DetectionEvent


class DetectionEventHandler:
    """Handles detection events by saving images to disk"""

    def __init__(self, config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

        # Subscribe to detection events
        self.event_bus.subscribe(DetectionEvent, self._on_detection)

    async def _on_detection(self, event: DetectionEvent):
        """Handle detection events by saving images"""
        # Save detection image
        self._save_image(event.frame, event.confidence, event.timestamp)

    def _save_image(self, frame, confidence, timestamp):
        """Save detection image to disk"""
        try:
            if not self.config.save_detections:
                return

            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"pigeon_detection_{timestamp_str}_{confidence:.3f}.{self.config.image_format}"
            filepath = Path(self.config.detection_folder) / filename

            # Save with specified quality
            if self.config.image_format.lower() == 'jpg':
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality])
            else:
                cv2.imwrite(str(filepath), frame)

            self.logger.info(f"Detection image saved: {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving detection image: {e}")
