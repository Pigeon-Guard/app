"""Event system for decoupling components"""
from .event_bus import EventBus
from .detection_event import DetectionEvent
from .frame_event import FrameEvent

__all__ = ['EventBus', 'DetectionEvent', 'FrameEvent']
