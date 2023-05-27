import asyncio
from typing import List

import cv2
import numpy as np
import mediapipe as mp
from dotenv import dotenv_values

from lotpose.dtos.frame_dto import FrameDto
from lotpose.dtos.mono_result_dto import MonoResultDto
from utils.mediapipe_utils import draw_landmarks_on_image

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

env_vars = dotenv_values()
pose_landmarker_path = env_vars['pose_landmarker_path']


class MonoCamPoseLandmarker:
    """Single camera pose estimation"""

    _landmarkers: dict[int, PoseLandmarker]  # list of landmarkers for each camera

    # _single_results
    _current_mono_results: dict[int, MonoResultDto]

    def __init__(self, device_indices: List[int]):
        """

        :param num_cameras: the number of cameras input to the system
        """
        self._landmarkers = {
            device_idx: PoseLandmarker.create_from_options(
                PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=pose_landmarker_path),
                    running_mode=VisionRunningMode.VIDEO)
            )
            for device_idx in device_indices
        }

        self._current_mono_results = dict()

    async def process_async(self, frames: dict[int, FrameDto]) -> dict[int, MonoResultDto]:
        """process a batch of frames
        :param frames:  for each camera
        :rtype: pose landmarks in shape (33, 3)
        """
        self._current_mono_results = dict()
        mp_images = {device_idx: (mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(frame.value, cv2.COLOR_BGR2RGB)), frame.timestamp) for
                     device_idx, frame in frames.items()}

        # process images
        for device_idx in self._landmarkers.keys():
            img, ts = mp_images[device_idx]
            result = self._landmarkers[device_idx].detect_for_video(img, ts)
            annotated_img = cv2.cvtColor(draw_landmarks_on_image(img.numpy_view(), result), cv2.COLOR_RGB2BGR)
            self._current_mono_results[device_idx] = MonoResultDto(device_idx, result, img, annotated_img, ts)

        return self._current_mono_results


# if __name__ == '__main__':
#     mono_landmarker = MonoCamPoseLandmarker()
