"""Unit tests for EventBus"""
import unittest
import asyncio
from app.event import EventBus, FrameEvent, DetectionEvent
from datetime import datetime


class TestEventBus(unittest.TestCase):
    """Test cases for EventBus"""

    def setUp(self):
        """Set up test fixtures"""
        self.event_bus = EventBus()
        self.received_events = []

    def test_subscribe_and_publish_sync(self):
        """Test subscribing and publishing with synchronous handler"""
        def sync_handler(event):
            self.received_events.append(event)

        self.event_bus.subscribe(FrameEvent, sync_handler)

        event = FrameEvent(frame=None, frame_number=1, timestamp=datetime.now())
        asyncio.run(self.event_bus.publish(event))

        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0], event)

    def test_subscribe_and_publish_async(self):
        """Test subscribing and publishing with async handler"""
        async def async_handler(event):
            self.received_events.append(event)

        self.event_bus.subscribe(FrameEvent, async_handler)

        event = FrameEvent(frame=None, frame_number=1, timestamp=datetime.now())
        asyncio.run(self.event_bus.publish(event))

        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0], event)

    def test_multiple_subscribers(self):
        """Test multiple subscribers receive the same event"""
        handler1_events = []
        handler2_events = []

        def handler1(event):
            handler1_events.append(event)

        def handler2(event):
            handler2_events.append(event)

        self.event_bus.subscribe(FrameEvent, handler1)
        self.event_bus.subscribe(FrameEvent, handler2)

        event = FrameEvent(frame=None, frame_number=1, timestamp=datetime.now())
        asyncio.run(self.event_bus.publish(event))

        self.assertEqual(len(handler1_events), 1)
        self.assertEqual(len(handler2_events), 1)
        self.assertEqual(handler1_events[0], event)
        self.assertEqual(handler2_events[0], event)

    def test_different_event_types(self):
        """Test subscribers only receive events they subscribed to"""
        frame_events = []
        detection_events = []

        def frame_handler(event):
            frame_events.append(event)

        def detection_handler(event):
            detection_events.append(event)

        self.event_bus.subscribe(FrameEvent, frame_handler)
        self.event_bus.subscribe(DetectionEvent, detection_handler)

        frame_event = FrameEvent(frame=None, frame_number=1, timestamp=datetime.now())
        detection_event = DetectionEvent(
            frame=None, confidence=0.9, x=0, y=0, width=100, height=100,
            frame_number=1, timestamp=datetime.now()
        )

        asyncio.run(self.event_bus.publish(frame_event))
        asyncio.run(self.event_bus.publish(detection_event))

        self.assertEqual(len(frame_events), 1)
        self.assertEqual(len(detection_events), 1)
        self.assertIsInstance(frame_events[0], FrameEvent)
        self.assertIsInstance(detection_events[0], DetectionEvent)

    def test_unsubscribe(self):
        """Test unsubscribing from events"""
        def handler(event):
            self.received_events.append(event)

        self.event_bus.subscribe(FrameEvent, handler)
        self.event_bus.unsubscribe(FrameEvent, handler)

        event = FrameEvent(frame=None, frame_number=1, timestamp=datetime.now())
        asyncio.run(self.event_bus.publish(event))

        self.assertEqual(len(self.received_events), 0)

    def test_handler_exception_does_not_break_other_handlers(self):
        """Test that exception in one handler doesn't affect others"""
        handler2_events = []

        def failing_handler(event):
            raise ValueError("Handler error")

        def successful_handler(event):
            handler2_events.append(event)

        self.event_bus.subscribe(FrameEvent, failing_handler)
        self.event_bus.subscribe(FrameEvent, successful_handler)

        event = FrameEvent(frame=None, frame_number=1, timestamp=datetime.now())
        asyncio.run(self.event_bus.publish(event))

        # Second handler should still receive the event
        self.assertEqual(len(handler2_events), 1)

    def test_async_handler_execution(self):
        """Test async handlers are properly awaited"""
        async def async_handler(event):
            await asyncio.sleep(0.01)  # Simulate async work
            self.received_events.append(event)

        self.event_bus.subscribe(FrameEvent, async_handler)

        event = FrameEvent(frame=None, frame_number=1, timestamp=datetime.now())
        asyncio.run(self.event_bus.publish(event))

        self.assertEqual(len(self.received_events), 1)


if __name__ == '__main__':
    unittest.main()
