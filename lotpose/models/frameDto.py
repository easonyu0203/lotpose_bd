import time
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameDto:
    """represents a frame with a timestamp"""
    device_index: int
    value: np.ndarray
    timestamp: float = time.time()

