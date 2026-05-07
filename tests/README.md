# Pigeon Guard Unit Tests

Unit tests for the Pigeon Guard detection system.

## Test Coverage

- **test_event_bus.py**: Tests for the event bus system
  - Event publishing and subscription
  - Multiple subscribers
  - Event type filtering
  - Async/sync handler support
  - Error handling

- **test_event_handlers.py**: Tests for event handlers
  - Detection event handler (image saving)
  - Notification event handler (Pushover notifications)
  - Configuration handling
  - Error scenarios

- **test_stream_observer.py**: Tests for video stream observer
  - Stream connection
  - Frame event emission
  - Frame skipping
  - Reconnection logic
  - Clean shutdown

- **test_notifier.py**: Tests for Pushover notifier
  - Notification sending
  - Image attachments
  - Credential validation
  - Error handling
  - Message formatting

- **test_detector.py**: Tests for detector
  - Avoid events during warmup period
  - Avoid events during cooldown period
  - Fire event for each detection in case warmup and cooldown are deactivated
  - Figure out whether to fire an event for a detection while warmmup and cooldown are active

## Running Tests

Run all tests:
```bash
python3 tests/run_tests.py
```

Run specific test file:
```bash
python3 -m unittest tests.test_event_bus
```

Run specific test case:
```bash
python3 -m unittest tests.test_event_bus.TestEventBus.test_subscribe_and_publish_sync
```

Run with verbose output:
```bash
python3 -m unittest discover -v tests
```

## Test Structure

Tests use Python's built-in `unittest` framework and include:
- Mock objects for external dependencies (cv2, requests, file I/O)
- Async test support for event bus and handlers
- Temporary directories for file operations
- Comprehensive error scenario testing

## Requirements

All test dependencies are included in the main `requirements.txt`:
- unittest (built-in)
- unittest.mock (built-in)
- asyncio (built-in)
