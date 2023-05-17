from typing import List

from lotpose.frame_collector import FrameCollector
from lotpose.webcam_controller import WebcamController
from utils import cv_utils


class WebcamManager:
    _webcam_controllers: List[WebcamController]
    _frame_collector: FrameCollector

    def __init__(self, device_indices: List[int], frame_collector: FrameCollector,
                 request_width: int,
                 request_height: int):

        # make controllers
        self._webcam_controllers = [WebcamController(idx, request_width, request_height) for idx in device_indices]

        self._frame_collector = frame_collector

    def start_all(self):
        """start all webcams"""
        for webcam_ctr in self._webcam_controllers:
            webcam_ctr.start()

    def stop_all(self):
        """stop all webcams"""
        for webcam_ctr in self._webcam_controllers:
            webcam_ctr.stop()

    def get_frames(self):
        """get batch of frames from each source"""
        return self._frame_collector.get_frames(self._webcam_controllers)
