from dataclasses import dataclass

import numpy as np


@dataclass
class Landmark3dDto:
    """represents a result from a single camera"""
    device_index: int
    value: np.array
    timestamp: int
