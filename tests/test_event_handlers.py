"""Unit tests for event handlers"""
import unittest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import numpy as np

from app.event import EventBus, DetectionEvent
from app.event_handler import DetectionEventHandler, NotificationEventHandler


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


class TestDetectionEventHandler(unittest.TestCase):
    """Test cases for DetectionEventHandler"""

    def setUp(self):
        """Set up test fixtures"""
        self.event_bus = EventBus()
        self.config = MockConfig()
        self.handler = DetectionEventHandler(self.config, self.event_bus)

    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temp directory
        import shutil
        if os.path.exists(self.config.detection_folder):
            shutil.rmtree(self.config.detection_folder)

    @patch('cv2.imwrite')
    def test_saves_image_on_detection_event(self, mock_imwrite):
        """Test that images are saved when detection event is received"""
        mock_imwrite.return_value = True

        # Create fake frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        event = DetectionEvent(
            frame=frame,
            confidence=0.95,
            x=10,
            y=10,
            width=50,
            height=50,
            frame_number=100,
            timestamp=datetime.now()
        )

        asyncio.run(self.event_bus.publish(event))

        # Verify cv2.imwrite was called
        mock_imwrite.assert_called_once()
        call_args = mock_imwrite.call_args
        saved_path = call_args[0][0]

        # Check file path format
        self.assertIn('pigeon_detection_', saved_path)
        self.assertTrue(saved_path.endswith('.jpg'))

    def test_does_not_save_when_disabled(self):
        """Test that images are not saved when save_detections is False"""
        self.config.save_detections = False

        with patch('cv2.imwrite') as mock_imwrite:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            event = DetectionEvent(
                frame=frame,
                confidence=0.95,
                x=10,
                y=10,
                width=50,
                height=50,
                frame_number=100,
                timestamp=datetime.now()
            )

            asyncio.run(self.event_bus.publish(event))

            # Verify cv2.imwrite was not called
            mock_imwrite.assert_not_called()

    @patch('cv2.imwrite')
    def test_saves_with_correct_quality(self, mock_imwrite):
        """Test that JPEG quality parameter is passed correctly"""
        mock_imwrite.return_value = True

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        event = DetectionEvent(
            frame=frame,
            confidence=0.95,
            x=10,
            y=10,
            width=50,
            height=50,
            frame_number=100,
            timestamp=datetime.now()
        )

        asyncio.run(self.event_bus.publish(event))

        # Check quality parameter
        call_args = mock_imwrite.call_args
        self.assertEqual(call_args[0][2], [1, 85])  # cv2.IMWRITE_JPEG_QUALITY value


class TestNotificationEventHandler(unittest.TestCase):
    """Test cases for NotificationEventHandler"""

    def setUp(self):
        """Set up test fixtures"""
        self.event_bus = EventBus()
        self.config = MockConfig()

    @patch('app.notifier.Notifier.send_notification')
    def test_sends_notification_on_detection_event(self, mock_send):
        """Test that notification is sent when detection event is received"""
        self.handler = NotificationEventHandler(self.config, self.event_bus)

        event = DetectionEvent(
            frame=np.zeros((100, 100, 3), dtype=np.uint8),
            confidence=0.95,
            x=10,
            y=10,
            width=50,
            height=50,
            frame_number=100,
            timestamp=datetime.now()
        )

        asyncio.run(self.event_bus.publish(event))

        # Verify notification was sent
        mock_send.assert_called_once()
        call_args = mock_send.call_args[0][0]

        self.assertIn('timestamp', call_args)
        self.assertIn('confidence', call_args)
        self.assertEqual(call_args['confidence'], 0.95)

    @patch('app.notifier.Notifier.send_notification')
    def test_does_not_send_when_disabled(self, mock_send):
        """Test that notification is not sent when pushover is disabled"""
        self.config.pushover_enabled = False
        self.handler = NotificationEventHandler(self.config, self.event_bus)

        event = DetectionEvent(
            frame=np.zeros((100, 100, 3), dtype=np.uint8),
            confidence=0.95,
            x=10,
            y=10,
            width=50,
            height=50,
            frame_number=100,
            timestamp=datetime.now()
        )

        asyncio.run(self.event_bus.publish(event))

        # Verify notification was not sent
        mock_send.assert_not_called()

    @patch('app.notifier.Notifier.send_notification')
    def test_notification_includes_image_path(self, mock_send):
        """Test that notification includes the expected image path"""
        self.handler = NotificationEventHandler(self.config, self.event_bus)

        timestamp = datetime(2024, 1, 15, 10, 30, 45, 123000)
        event = DetectionEvent(
            frame=np.zeros((100, 100, 3), dtype=np.uint8),
            confidence=0.95,
            x=10,
            y=10,
            width=50,
            height=50,
            frame_number=100,
            timestamp=timestamp
        )

        asyncio.run(self.event_bus.publish(event))

        call_args = mock_send.call_args[0][0]
        image_path = call_args['image_path']

        # Verify image path format
        self.assertIn('pigeon_detection_20240115_103045_123_0.950', image_path)
        self.assertTrue(image_path.endswith('.jpg'))

    @patch('app.notifier.Notifier.send_notification')
    def test_handles_notification_error_gracefully(self, mock_send):
        """Test that notification errors are handled gracefully"""
        mock_send.side_effect = Exception("Network error")

        self.handler = NotificationEventHandler(self.config, self.event_bus)

        event = DetectionEvent(
            frame=np.zeros((100, 100, 3), dtype=np.uint8),
            confidence=0.95,
            x=10,
            y=10,
            width=50,
            height=50,
            frame_number=100,
            timestamp=datetime.now()
        )

        # Should not raise exception
        asyncio.run(self.event_bus.publish(event))


if __name__ == '__main__':
    unittest.main()
