from typing import Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DetectionEvent:
    """Event emitted when a pigeon is detected"""
    frame: Any
    confidence: float
    x: int
    y: int
    width: int
    height: int
    frame_number: int
    timestamp: datetime
