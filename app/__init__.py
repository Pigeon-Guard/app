"""Pigeon Detector"""

# Don't import Application and Config here to avoid importing all dependencies
# when importing submodules (e.g., from app.event import EventBus)
# Import them directly when needed: from app.application import Application

__all__ = ['application', 'config', 'event', 'detector', 'input', 'event_handler', 'notifier']