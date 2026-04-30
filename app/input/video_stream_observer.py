"""Stream observer for capturing video frames"""
import asyncio
import cv2
import logging
import time
from datetime import datetime
from app.event import EventBus, FrameEvent


class VideoStreamObserver:
    """Observes a video stream and emits frame events"""

    def __init__(self, config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.frame_count = 0
        self.cap = None

    def _connect(self):
        """Connect to video stream with retry logic"""
        attempt = 0
        while True:
            try:
                self.logger.info(f"Connecting to stream: {self.config.stream_url} (attempt {attempt + 1})")
                cap = cv2.VideoCapture(self.config.stream_url)

                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.logger.info("Successfully connected to stream")
                        return cap
                    else:
                        cap.release()
                        self.logger.warning("Stream connected but cannot read frames")
                else:
                    self.logger.warning("Failed to open stream")

            except Exception as e:
                self.logger.error(f"Stream connection error: {e}")

            if 0 <= self.config.reconnect_attempts <= attempt:
                break
            else:
                attempt += 1
                self.logger.info(f"Retrying in {self.config.reconnect_delay} seconds...")
                time.sleep(self.config.reconnect_delay)

        self.logger.error("Failed to connect to stream after all attempts")
        return None

    async def start(self):
        """Start observing the stream"""
        if self.running:
            self.logger.warning("Stream observer already running")
            return

        self.running = True
        self.cap = self._connect()

        if self.cap is None:
            self.logger.error("Could not connect to stream")
            self.running = False
            return

        self.logger.info("Stream observer started")

        try:
            while self.running:
                ret, frame = self.cap.read()

                if not ret:
                    self.logger.warning("Failed to read frame from stream")
                    self.cap.release()
                    self.cap = self._connect()

                    if self.cap is None:
                        self.logger.error("Stream reconnection failed")
                        break
                    continue

                self.frame_count += 1

                # Skip frames for performance
                if self.frame_count % self.config.frame_skip != 0:
                    continue

                # Emit frame event
                event = FrameEvent(
                    frame=frame,
                    frame_number=self.frame_count,
                    timestamp=datetime.now()
                )
                await self.event_bus.publish(event)

                # Log progress periodically
                if self.frame_count % 1000 == 0:
                    self.logger.info(f"Processed {self.frame_count} frames")

                # Small delay to allow other tasks to run
                await asyncio.sleep(0)

        except Exception as e:
            self.logger.error(f"Stream processing error: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.logger.info("Stream observer stopped")

    def stop(self):
        """Stop observing the stream"""
        self.logger.info("Stopping stream observer...")
        self.running = False
