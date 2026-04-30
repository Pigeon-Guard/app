"""Pigeon detection component"""
import logging
import time
from ultralytics import YOLO

from app.event import EventBus, FrameEvent, DetectionEvent
# from detector.hailo import HailoHEFModel

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
            if self.config.model_path.endswith(".pt"):
                self.model = YOLO(self.config.model_path)
            elif self.config.model_path.endswith(".hef"):
                raise NotImplementedError("*.hef support is not implemented yet")
                # TODO: self.model = HailoHEFModel(self.config.model_path)
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
            confidence, x, y, width, height = self.model.predict(event.frame)
            is_pigeon = confidence > self.config.confidence_threshold

            if self.config.debug_mode:
                self.logger.debug(f"Detection confidence: {confidence:.4f}, threshold: {self.config.confidence_threshold}")

            # Check cooldown and emit detection event if pigeon detected
            if is_pigeon:
                current_time = time.time()

                # Check cooldown period
                if current_time - self.last_detection_time < self.config.cooldown_period:
                    if self.config.debug_mode:
                        self.logger.debug("Detection in cooldown period, skipping event")
                    return

                self.last_detection_time = current_time
                self.detection_count += 1

                self.logger.info(
                    f"Pigeon detected! Confidence: {confidence:.2%}, "
                    f"Total detections: {self.detection_count}"
                )

                detection_event = DetectionEvent(
                    frame=event.frame,
                    confidence=confidence,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    frame_number=event.frame_number,
                    timestamp=event.timestamp
                )
                await self.event_bus.publish(detection_event)

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
