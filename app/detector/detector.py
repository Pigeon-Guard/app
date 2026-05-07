"""Pigeon detection component"""
import logging
import time
from datetime import datetime
from typing import Optional
from app.event import EventBus, FrameEvent, DetectionEvent

class Detector:
    """Detects pigeons in frames and emits detection events"""

    def __init__(self, config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.cooldown_end_time = 0
        self.warmup_end_time = 0
        self.detection_count = 0

        # Subscribe to frame events
        self.event_bus.subscribe(FrameEvent, self._on_frame)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the pigeon detection model"""
        try:
            if self.config.model_path.endswith(".hef"):
                from app.hailo import HEFModel
                self.model = HEFModel(self.config.model_path)

            elif self.config.model_path.endswith(".pt"):
                from ultralytics import YOLO
                self.model = YOLO(self.config.model_path)

            else:
                raise ValueError(f"Invalid model file format: {self.config.model_path}")
            self.logger.info(f"Model loaded: {self.config.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    async def _on_frame(self, event: FrameEvent):
        """Handle incoming frame events"""
        try:
            # Run detection
            detection = await self.detect(event.frame, event.frame_number, save_image=self.config.save_detections)

            if detection:
                current_time = time.time()

                # Check cooldown period
                if current_time <= self.cooldown_end_time:
                    self.logger.debug("Detection in cooldown period, skipping event")
                    return

                if self.config.warmup_seconds > 0:
                    # Check warmup period
                    if current_time < self.warmup_end_time:
                        self.logger.debug("Consecutive detection in warmup period, skipping event")
                        return

                    # Reset warmup end time if current detection is not consecutive
                    if self.warmup_end_time < current_time - self.config.warmup_seconds:
                        self.logger.debug("New detection starting warmup period, skipping event")
                        self.warmup_end_time = current_time + self.config.warmup_seconds
                        return

                self.warmup_end_time = current_time + self.config.warmup_seconds
                self.cooldown_end_time = current_time + self.config.cooldown_seconds
                self.detection_count += 1

                self.logger.info(
                    f"Pigeon detected! Confidence: {detection.confidence:.2%}, "
                    f"Total detections: {self.detection_count}"
                )

                await self.event_bus.publish(detection)

        except Exception as e:
            self.logger.error(f"Detection error: {e}")

    async def detect(self, frame, frame_number: int = 0, save_image: bool = True) -> Optional[DetectionEvent]:
        """
        Run detection on a single frame/image.

        Args:
            frame: The image frame to analyze
            frame_number: Optional frame number for tracking
            save_image: Whether to save the detection image (default: True)

        Returns:
            dict with keys:
                - is_pigeon: bool
                - confidence: float
                - x, y, width, height: int (bounding box)
                - detection_event: DetectionEvent if pigeon detected, else None
        """
        try:
            # Run detection
            confidence, x, y, width, height = 0.0, -1, -1, -1, -1

            results = self.model.predict(frame)
            if results and len(results):
                result = results[0]

                if result['bbox']:
                    # Hailo yolov8n
                    confidence = result['confidence']
                    x, y, width, height = result['bbox']
                
                elif result.boxes:
                    # ultralytics YOLO
                    box = result.boxes[0]
                    confidence = box.conf[0]
                    x, y, width, height = box.xywh[0]

            is_pigeon = confidence > self.config.confidence_threshold

            self.logger.debug(f"Detection confidence: {confidence:.4f}, threshold: {self.config.confidence_threshold}")

            detection_event = None
            if is_pigeon:
                timestamp = datetime.now()

                detection_event = DetectionEvent(
                    frame=frame,
                    confidence=confidence,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    save_image=save_image
                )

            return detection_event

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            raise
