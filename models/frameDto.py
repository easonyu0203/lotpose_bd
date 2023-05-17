from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class FrameDto:
    """represents a frame with a timestamp"""
    device_index: int
    value: np.ndarray
    timestamp: datetime = datetime.now()

