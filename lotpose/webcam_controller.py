from typing import Tuple, Optional

import cv2
import numpy as np

from lotpose.dtos.frame_dto import FrameDto


class WebcamController:
    """act like a controller for a webcam"""
    device_index: int
    width: int
    height: int
    mtx: Optional[np.array]
    dist: Optional[np.array]
    new_mtx: Optional[np.array]
    is_calibrated: bool

    def __init__(self, device_index: int, request_width: int, request_height: int):
        """
        :param device_index: the device index of the webcam
        """
        self.device_index = device_index
        self.width, self.height = self._get_width_height(request_width, request_height)
        self._capture = None
        self.is_calibrated = False

    def start(self):
        """start the webcam and put the frames in the queue"""
        self._capture = cv2.VideoCapture(self.device_index)

        # Set the webcam resolution
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def stop(self):
        """stop the webcam"""
        self._capture.release()

    def get_frame(self) -> FrameDto:
        """get a frame from the queue"""

        # Capture frame-by-frame
        _, frame = self._capture.read()

        # undistort the frame
        # if self.is_calibrated:
        #     frame = cv2.undistort(frame, self.mtx, self.dist, None, self.new_mtx)

        return FrameDto(self.device_index, frame)

    def _get_width_height(self, request_width: int, request_height: int) -> Tuple[int, int]:
        """request width and height from the webcam"""
        capture = cv2.VideoCapture(self.device_index)

        # Set the webcam resolution
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, request_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, request_height)

        # Get the actual width and height from the frame
        _, frame = capture.read()
        return frame.shape[1], frame.shape[0]

    def set_calibrate_data(self, mtx: np.array, dist: np.array):
        self.mtx = mtx
        self.dist = dist
        self.new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.width, self.height), 0, (self.width, self.height))
        self.is_calibrated = True
