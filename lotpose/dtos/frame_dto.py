import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameDto:
    """represents a frame with a timestamp"""
    device_index: int
    value: np.ndarray
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))  # (ms)
