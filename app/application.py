"""Main application orchestrator"""
import logging
import signal
import sys
from typing import Optional

from app.event import EventBus, DetectionEvent
from app.input import VideoStreamObserver
from app.detector import Detector
from app.event_handler import DetectionEventHandler, NotificationEventHandler

class Application:
    """Main application that orchestrates all components"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = False

        # Create event bus
        self.event_bus = EventBus()

        # Create components
        self.stream_observer = VideoStreamObserver(config, self.event_bus)
        self.detector = Detector(config, self.event_bus)
        self.detection_handler = DetectionEventHandler(config, self.event_bus)
        self.notification_handler = NotificationEventHandler(config, self.event_bus)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("Application initialized")

    async def start(self):
        """Start the application"""
        if self.running:
            self.logger.warning("Application already running")
            return

        self.running = True
        self.logger.info("Starting application...")

        # Start stream observer
        await self.stream_observer.start()

    def stop(self):
        """Stop the application"""
        if not self.running:
            return

        self.logger.info("Stopping application...")
        self.running = False
        self.stream_observer.stop()
        self.logger.info("Application stopped")

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def get_status(self):
        """Get application status"""
        return {
            'running': self.running,
            'frames_processed': self.stream_observer.frame_count,
            'detections_count': self.detector.detection_count,
            'stream_url': self.config.stream_url,
            'model_path': self.config.model_path
        }

    async def detect(self, frame) -> Optional[DetectionEvent]:
        """
        Run detection on a single image.

        Args:
            frame: The image frame to analyze

        Returns:
            dict with keys:
                - is_pigeon: bool
                - confidence: float
        """
        # Don't save single image detections to disk
        return await self.detector.detect(frame, save_image=False)
