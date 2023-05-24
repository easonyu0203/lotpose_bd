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
                    running_mode=VisionRunningMode.LIVE_STREAM,
                    result_callback=lambda r, img, ts: self._result_callback(device_idx, r, img, ts))
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
        mp_images = {device_idx: (mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.value), frame.timestamp) for
                     device_idx, frame in frames.items()}

        # process images
        for device_idx in self._landmarkers.keys():
            img, ts = mp_images[device_idx]
            self._landmarkers[device_idx].detect_async(img, ts)

        # wait for results
        while True:
            if len(self._current_mono_results) != len(self._landmarkers):
                await asyncio.sleep(0.01)
                continue

            return self._current_mono_results

    def _result_callback(self, idx: int, result: PoseLandmarkerResult, input_image: mp.Image, timestamp_ms: int):
        annotated_img = draw_landmarks_on_image(input_image.numpy_view(), result)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        self._current_mono_results[idx] = MonoResultDto(idx, result, input_image, annotated_img, timestamp_ms)


# if __name__ == '__main__':
#     mono_landmarker = MonoCamPoseLandmarker()
