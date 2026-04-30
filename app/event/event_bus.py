import asyncio
import logging
from typing import Any, Callable, Dict

class EventBus:
    """Simple async event bus for publishing and subscribing to events"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.subscribers: Dict[type, list[Callable]] = {}

    def subscribe(self, event_type: type, handler: Callable):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        self.logger.debug(f"Subscribed {handler.__name__} to {event_type.__name__}")

    def unsubscribe(self, event_type: type, handler: Callable):
        """Unsubscribe from an event type"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)

    async def publish(self, event: Any):
        """Publish an event to all subscribers"""
        event_type = type(event)
        if event_type in self.subscribers:
            tasks = []
            for handler in self.subscribers[event_type]:
                # Create task for async handlers, call sync handlers directly
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(asyncio.create_task(handler(event)))
                else:
                    try:
                        handler(event)
                    except Exception as e:
                        self.logger.error(f"Error in sync handler {handler.__name__}: {e}")

            # Wait for all async tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
