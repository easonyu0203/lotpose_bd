from dataclasses import dataclass

import numpy as np
from mediapipe.tasks.python.vision import PoseLandmarkerResult
import mediapipe as mp


@dataclass
class MonoResultDto:
    """represents a result from a single camera"""
    device_index: int
    result: PoseLandmarkerResult
    input_img: mp.Image
    annotated_img: np.array  # BGR image
    timestamp: int


