from typing import List

from frame_collector import FrameCollector
from utils import cv_utils
from webcam_controller import WebcamController


class WebcamManager:
    webcam_controllers: List[WebcamController]
    frame_collector: FrameCollector

    def __init__(self, device_indices: List[int], frame_collector: FrameCollector):

        # make controllers
        self.webcam_controllers = [WebcamController(idx) for idx in device_indices]

        self.frame_collector = frame_collector

    def start_all(self):
        """start all webcams"""
        for webcam_ctr in self.webcam_controllers:
            webcam_ctr.start()

    def stop_all(self):
        """stop all webcams"""
        for webcam_ctr in self.webcam_controllers:
            webcam_ctr.stop()

    def get_frames(self):
        """get batch of frames from each source"""
        return self.frame_collector.get_frames(self.webcam_controllers)
