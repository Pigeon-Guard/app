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
        self.last_detection_time = 0
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

    # def _preprocess_frame(self, frame):
    #     """Preprocess frame for model input"""
    #     try:
    #         # Resize frame
    #         frame_resized = cv2.resize(frame, (self.config.input_width, self.config.input_height))
    #
    #         # Convert BGR to RGB if needed
    #         if self.config.bgr_to_rgb:
    #             frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    #         else:
    #             frame_rgb = frame_resized
    #
    #         # Normalize if needed
    #         if self.config.normalize:
    #             frame_normalized = frame_rgb.astype(np.float32) / 255.0
    #         else:
    #             frame_normalized = frame_rgb.astype(np.float32)
    #
    #         return frame_normalized
    #
    #     except Exception as e:
    #         self.logger.error(f"Frame preprocessing error: {e}")
    #         return None

    async def _on_frame(self, event: FrameEvent):
        """Handle incoming frame events"""
        try:
            # TODO: Check if we need any preprocessing at all
            # Preprocess frame
            # frame_preprocessed = self._preprocess_frame(event.frame)
            # if frame_preprocessed is None:
            #     return

            # Run detection
            detection = await self.detect(event.frame, event.frame_number, save_image=self.config.save_detections)

            if detection:
                current_time = time.time()

                # Check cooldown period
                if current_time - self.last_detection_time < self.config.cooldown_period:
                    if self.config.debug_mode:
                        self.logger.debug("Detection in cooldown period, skipping event")
                    return

                self.last_detection_time = current_time
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

            # TODO: ultralytics YOLO specific
            results = self.model.predict(frame)
            if results and len(results) and len(results[0].boxes):
                box = results[0].boxes[0]
                confidence = box.conf[0]
                x, y, width, height = box.xywh[0]

            is_pigeon = confidence > self.config.confidence_threshold

            if self.config.debug_mode:
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
