from typing import Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FrameEvent:
    """Event emitted when a new frame is available"""
    frame: Any
    frame_number: int
    timestamp: datetime
