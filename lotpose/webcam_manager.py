from typing import List

import numpy as np

from lotpose.frame_collector import FrameCollector
from lotpose.dtos.frame_dto import FrameDto
from lotpose.webcam_controller import WebcamController


class WebcamManager:
    _webcam_controllers: dict[int, WebcamController]
    _frame_collector: FrameCollector
    _webcam_pair_RT: dict[tuple[int, int], tuple[np.array, np.array]]

    def __init__(self, device_indices: List[int], frame_collector: FrameCollector,
                 request_width: int,
                 request_height: int):

        # make controllers
        self._webcam_controllers = {idx: WebcamController(idx, request_width, request_height) for idx in device_indices}

        self._frame_collector = frame_collector

        self._webcam_pair_RT = dict()

    def start_all(self):
        """start all webcams"""
        for webcam_ctr in self._webcam_controllers.values():
            webcam_ctr.start()

    def stop_all(self):
        """stop all webcams"""
        for webcam_ctr in self._webcam_controllers.values():
            webcam_ctr.stop()

    def get_frames(self) -> dict[int, FrameDto]:
        """get batch of frames from each source"""
        batch_frames = self._frame_collector.get_frames(self._webcam_controllers)
        return batch_frames

    def __getitem__(self, index: int):
        """get individual webcam controller"""
        return self._webcam_controllers[index]

    def set_calibrate_data(self, cam1_idx, cam2_idx, k1, d1, k2, d2, r, t):
        self._webcam_pair_RT[(cam1_idx, cam2_idx)] = (r, t)
        self._webcam_controllers[cam1_idx].set_calibrate_data(k1, d1)
        self._webcam_controllers[cam1_idx].set_calibrate_data(k2, d2)

    # def to_other_camera_coordinate(self, from_cam_idx, to_cam_idx, norm_):