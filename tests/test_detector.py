"""Unit tests for detetor"""
import unittest
import tempfile
import numpy as np

from freezegun import freeze_time
from datetime import datetime
from unittest.mock import patch, create_autospec

from app.event import EventBus, DetectionEvent, FrameEvent
from app.detector import Detector

class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.save_detections = True
        self.detection_folder = tempfile.mkdtemp()
        self.image_format = 'jpg'
        self.image_quality = 85
        self.pushover_enabled = True
        self.pushover_user_key = 'test_user_key'
        self.pushover_api_token = 'test_api_token'

class TestDetector(unittest.IsolatedAsyncioTestCase):
    """Test cases for Detector"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.event_bus = create_autospec(EventBus)
        self.config = MockConfig()

    @patch.object(Detector, "_load_model")
    async def test_publishes_event_for_each_detection(self, mock_method):
        """Verify publishing event for each detection if warmup and cooldown are deactivated"""
        detector = Detector(self.config, self.event_bus)
        frame_event = FrameEvent(
            frame=np.zeros((640, 640, 3), dtype=np.uint8),
            frame_number=100,
            timestamp=datetime.now()
        )
        detection_event = DetectionEvent(
            frame=frame_event.frame,
            confidence=0.95,
            x=10,
            y=10,
            width=50,
            height=50,
            frame_number=100,
            timestamp=datetime.now()
        )

        self.config.warmup_seconds = 0
        self.config.cooldown_seconds = 0
        sim_detection_count = 6

        with patch.object(detector, 'detect', return_value=detection_event) as mock_detect:
            self.assertEqual(detector.detection_count, 0)

            with freeze_time("2026-05-07 10:00:00") as frozen_time:
                for i in range(sim_detection_count):
                    frozen_time.tick()

                    frame_event.frame_number = i
                    frame_event.timestamp = datetime.now()
                    await detector._on_frame(frame_event)

            self.assertEqual(sim_detection_count, mock_detect.call_count)
            self.assertEqual(sim_detection_count, detector.detection_count)
            self.assertEqual(sim_detection_count, self.event_bus.publish.call_count)

    @patch.object(Detector, "_load_model")
    async def test_publishes_event_for_each_first_detection_after_warmup(self, mock_method):
        """Verify only first detection after warmup fires an event"""
        detector = Detector(self.config, self.event_bus)
        frame_event = FrameEvent(
            frame=np.zeros((640, 640, 3), dtype=np.uint8),
            frame_number=100,
            timestamp=datetime.now()
        )
        detection_event = DetectionEvent(
            frame=frame_event.frame,
            confidence=0.95,
            x=10,
            y=10,
            width=50,
            height=50,
            frame_number=100,
            timestamp=datetime.now()
        )

        self.config.warmup_seconds = 2
        self.config.cooldown_seconds = 0
        sim_detection_count = 6

        with patch.object(detector, 'detect', return_value=detection_event) as mock_detect:
            self.assertEqual(detector.detection_count, 0)

            with freeze_time("2026-05-07 10:00:00") as frozen_time:
                for i in range(sim_detection_count):
                    frozen_time.tick()

                    frame_event.frame_number = i
                    frame_event.timestamp = datetime.now()
                    await detector._on_frame(frame_event)

            self.assertEqual(sim_detection_count, mock_detect.call_count)
            self.assertEqual(2, detector.detection_count)
            self.assertEqual(2, self.event_bus.publish.call_count)

    @patch.object(Detector, "_load_model")
    async def test_publishes_event_for_each_first_detection_after_cooldown(self, mock_method):
        """Verify no events are fired during cooldown"""
        detector = Detector(self.config, self.event_bus)
        frame_event = FrameEvent(
            frame=np.zeros((640, 640, 3), dtype=np.uint8),
            frame_number=100,
            timestamp=datetime.now()
        )
        detection_event = DetectionEvent(
            frame=frame_event.frame,
            confidence=0.95,
            x=10,
            y=10,
            width=50,
            height=50,
            frame_number=100,
            timestamp=datetime.now()
        )

        self.config.warmup_seconds = 0
        self.config.cooldown_seconds = 1
        sim_detection_count = 6

        with patch.object(detector, 'detect', return_value=detection_event) as mock_detect:
            self.assertEqual(detector.detection_count, 0)

            with freeze_time("2026-05-07 10:00:00") as frozen_time:
                for i in range(sim_detection_count):
                    frozen_time.tick()

                    frame_event.frame_number = i
                    frame_event.timestamp = datetime.now()
                    await detector._on_frame(frame_event)

            self.assertEqual(sim_detection_count, mock_detect.call_count)
            self.assertEqual(3, detector.detection_count)
            self.assertEqual(3, self.event_bus.publish.call_count)

    @patch.object(Detector, "_load_model")
    async def test_not_publishes_events_during_warmup_or_cooldown(self, mock_method):
        """Verify no event is fired during warmup or cooldown phase"""
        detector = Detector(self.config, self.event_bus)
        frame_event = FrameEvent(
            frame=np.zeros((640, 640, 3), dtype=np.uint8),
            frame_number=100,
            timestamp=datetime.now()
        )
        detection_event = DetectionEvent(
            frame=frame_event.frame,
            confidence=0.95,
            x=10,
            y=10,
            width=50,
            height=50,
            frame_number=100,
            timestamp=datetime.now()
        )

        self.config.warmup_seconds = 1
        self.config.cooldown_seconds = 1
        sim_detection_count = 6

        with patch.object(detector, 'detect', return_value=detection_event) as mock_detect:
            self.assertEqual(detector.detection_count, 0)

            with freeze_time("2026-05-07 10:00:00") as frozen_time:
                for i in range(sim_detection_count):
                    frozen_time.tick()

                    frame_event.frame_number = i
                    frame_event.timestamp = datetime.now()
                    await detector._on_frame(frame_event)

            self.assertEqual(sim_detection_count, mock_detect.call_count)
            self.assertEqual(3, detector.detection_count)
            self.assertEqual(3, self.event_bus.publish.call_count)
