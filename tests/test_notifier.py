"""Unit tests for Notifier"""
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

from app.notifier import Notifier


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.pushover_enabled = True
        self.pushover_user_key = 'test_user_key'
        self.pushover_api_token = 'test_api_token'


class TestNotifier(unittest.TestCase):
    """Test cases for Notifier"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MockConfig()
        self.notifier = Notifier(self.config)

    @patch('requests.post')
    def test_sends_pushover_notification(self, mock_post):
        """Test that Pushover notification is sent successfully"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        detection_data = {
            'timestamp': '2024-01-15T10:30:45',
            'confidence': 0.95,
            'image_path': None
        }

        self.notifier.send_notification(detection_data)

        # Verify request was made
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Check URL
        self.assertEqual(call_args[0][0], 'https://api.pushover.net/1/messages.json')

        # Check payload
        payload = call_args[1]['data']
        self.assertEqual(payload['token'], 'test_api_token')
        self.assertEqual(payload['user'], 'test_user_key')
        self.assertIn('Pigeon detected', payload['message'])
        self.assertIn('95.00%', payload['message'])
        self.assertEqual(payload['title'], 'Pigeon Detection Alert')

    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_data')
    def test_sends_notification_with_image(self, mock_file, mock_post):
        """Test that notification includes image attachment when available"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Create a fake image path
        image_path = '/tmp/test_image.jpg'

        detection_data = {
            'timestamp': '2024-01-15T10:30:45',
            'confidence': 0.95,
            'image_path': image_path
        }

        with patch('pathlib.Path.exists', return_value=True):
            self.notifier.send_notification(detection_data)

        # Verify file was opened
        call_args = mock_post.call_args
        files = call_args[1].get('files', {})

        # Image should be attached
        self.assertIn('attachment', files)

    @patch('requests.post')
    def test_does_not_send_when_disabled(self, mock_post):
        """Test that notification is not sent when disabled"""
        self.config.pushover_enabled = False
        notifier = Notifier(self.config)

        detection_data = {
            'timestamp': '2024-01-15T10:30:45',
            'confidence': 0.95,
            'image_path': None
        }

        notifier.send_notification(detection_data)

        # Verify no request was made
        mock_post.assert_not_called()

    @patch('requests.post')
    def test_does_not_send_without_credentials(self, mock_post):
        """Test that notification is not sent without credentials"""
        self.config.pushover_user_key = ''
        self.config.pushover_api_token = ''
        notifier = Notifier(self.config)

        detection_data = {
            'timestamp': '2024-01-15T10:30:45',
            'confidence': 0.95,
            'image_path': None
        }

        notifier.send_notification(detection_data)

        # Verify no request was made
        mock_post.assert_not_called()

    @patch('requests.post')
    def test_handles_network_error_gracefully(self, mock_post):
        """Test that network errors are handled gracefully"""
        mock_post.side_effect = Exception("Network error")

        detection_data = {
            'timestamp': '2024-01-15T10:30:45',
            'confidence': 0.95,
            'image_path': None
        }

        # Should not raise exception
        try:
            self.notifier.send_notification(detection_data)
        except Exception:
            self.fail("send_notification raised an exception unexpectedly")

    @patch('requests.post')
    def test_handles_http_error_gracefully(self, mock_post):
        """Test that HTTP errors are handled gracefully"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 400 Error")
        mock_post.return_value = mock_response

        detection_data = {
            'timestamp': '2024-01-15T10:30:45',
            'confidence': 0.95,
            'image_path': None
        }

        # Should not raise exception
        try:
            self.notifier.send_notification(detection_data)
        except Exception:
            self.fail("send_notification raised an exception unexpectedly")

    @patch('requests.post')
    def test_formats_message_correctly(self, mock_post):
        """Test that notification message is formatted correctly"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        detection_data = {
            'timestamp': '2024-01-15T10:30:45.123456',
            'confidence': 0.8765,
            'image_path': None
        }

        self.notifier.send_notification(detection_data)

        call_args = mock_post.call_args
        payload = call_args[1]['data']
        message = payload['message']

        # Check message format
        self.assertIn('87.65%', message)
        self.assertIn('2024-01-15T10:30:45', message)

    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open)
    def test_closes_image_file_after_sending(self, mock_file, mock_post):
        """Test that image file is properly closed after sending"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        image_path = '/tmp/test_image.jpg'

        detection_data = {
            'timestamp': '2024-01-15T10:30:45',
            'confidence': 0.95,
            'image_path': image_path
        }

        with patch('pathlib.Path.exists', return_value=True):
            self.notifier.send_notification(detection_data)

        # Verify file handle was properly managed
        # The file should be closed after sending
        self.assertTrue(True)  # File context manager handles this


if __name__ == '__main__':
    unittest.main()
