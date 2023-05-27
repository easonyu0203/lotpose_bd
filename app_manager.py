import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Protocol
from itertools import combinations

import cv2
import numpy as np

from lotpose.dtos.mono_result_dto import MonoResultDto
from lotpose.frame_collector import FrameCollector
from lotpose.dtos.frame_dto import FrameDto
from lotpose.monocam_pose_landmarker import MonoCamPoseLandmarker
from lotpose.webcam_manager import WebcamManager
from utils import cv_utils

# settings
frame_collector_tolerant_interval = 30
request_width = 640
request_height = 480


@dataclass
class AppState:
    """State of the app"""
    webcam_stared: bool = False
    stared_device_indices: List[int] = None
    webcams_info: List[cv_utils.WebcamDeviceInfo] = None
    current_frames: dict[int, FrameDto] = field(default=None, repr=False)
    current_mono_results: dict[int, MonoResultDto] = field(default=None, repr=False)
    is_camera_calibrated: bool = False
    is_camera_calibrating: bool = False


@dataclass
class AppStateDto:
    webcam_stared: bool = False
    stared_device_indices: List[int] = None
    webcams_info: List[cv_utils.WebcamDeviceInfo] = None
    is_camera_calibrated: bool = False
    is_camera_calibrating: bool = False


class IAppManager(Protocol):
    _app_state: AppState

    def get_app_state_dto(self) -> AppStateDto:
        ...

    def start_webcams(self, device_indices: List[int]) -> None:
        ...

    async def start_calibration_bg_task(self) -> None:
        ...

    async def start_pipeline_bg_task(self) -> None:
        ...

    def get_webcams_frames(self) -> dict[int, FrameDto]:
        ...

    def get_mono_results(self) -> dict[int, MonoResultDto]:
        ...

    def stop_webcams_n_pipeline(self) -> None:
        ...


class AppManager(IAppManager):
    """Singleton class for managing the app"""
    Singleton: IAppManager = None
    _app_state: AppState
    webcam_manager: Optional[WebcamManager] = None
    mono_landmarker: MonoCamPoseLandmarker = None
    pipe_task: asyncio.Task = None

    def __init__(self):
        self._app_state = AppState()
        self._app_state.webcams_info = cv_utils.list_webcams()
        self._app_state.stared_device_indices = []

    def get_app_state_dto(self) -> AppStateDto:
        dto = AppStateDto(
            webcam_stared=self._app_state.webcam_stared,
            stared_device_indices=self._app_state.stared_device_indices,
            webcams_info=self._app_state.webcams_info,
            is_camera_calibrated=self._app_state.is_camera_calibrated,
            is_camera_calibrating=self._app_state.is_camera_calibrating
        )
        return dto

    def start_webcams(self, device_indices: List[int]) -> None:
        """init and start webcams"""

        assert self.webcam_manager is None, "Webcams already started"

        # init
        frame_collector = FrameCollector(tolerant_interval=frame_collector_tolerant_interval)
        self.webcam_manager = WebcamManager(device_indices, frame_collector, request_width, request_height)

        self.mono_landmarker = MonoCamPoseLandmarker(device_indices)

        # start webcams
        self.webcam_manager.start_all()
        self._app_state.webcam_stared = True
        self._app_state.stared_device_indices = device_indices

    async def start_pipeline_bg_task(self) -> None:
        """start process of input image and output prediction"""
        old_frames: Optional[dict[int, FrameDto]] = None
        while self.webcam_manager is not None:
            frames = self.webcam_manager.get_frames()
            # skip if no new frames
            if old_frames is not None and min(f.timestamp for f in frames.values()) == min(
                    f.timestamp for f in old_frames.values()):
                await asyncio.sleep(0.001)
                continue
            old_frames = frames

            # mono camera pose estimation pass
            mono_results = await self.mono_landmarker.process_async(frames)

            self._app_state.current_frames = frames
            self._app_state.current_mono_results = mono_results

            await asyncio.sleep(0.016)  # run 60 fps

    async def start_calibration_bg_task(self) -> None:
        """calibrate cameras"""
        assert self.webcam_manager is not None, "Webcams not started"
        assert len(self._app_state.stared_device_indices) >= 2, "At least 2 cameras needed"
        # change app state
        self._app_state.is_camera_calibrated = False
        self._app_state.is_camera_calibrating = True

        # capture chessboard images

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # (points_per_row,points_per_colum)
        points_per_row = 9
        points_per_column = 6
        pattern_size = (points_per_row, points_per_column)
        calibrate_sample_size = 10
        calibrate_progress = 0

        # Prepare object points
        objp = np.zeros((points_per_row * points_per_column, 3), np.float32)
        objp[:, :2] = np.mgrid[0:points_per_row, 0:points_per_column].T.reshape(-1, 2)

        # set up variable
        camera_pairs = list(combinations(self._app_state.stared_device_indices, 2))
        calibrate_data = {k: ([], [], []) for k in camera_pairs}  # (objpoints, imgpoints1, imgpoints2)
        # objpoints = []  # 3D points in real-world space
        # imgpoints1 = []  # 2D points in image plane for camera 1
        # imgpoints2 = []  # 2D points in image plane for camera 2

        while self.webcam_manager is not None:
            # check if stop
            if all(len(v[0]) >= calibrate_sample_size for v in calibrate_data.values()):
                break

            calibrate_progress = sum(min(len(v[0]), calibrate_sample_size) for v in calibrate_data.values()) / (
                        len(camera_pairs) * calibrate_sample_size)

            # get frames
            frames = self.webcam_manager.get_frames()
            gray_frames = {k: cv2.cvtColor(f.value, cv2.COLOR_BGR2GRAY) for k, f in frames.items()}

            # find the chessboard corners (rect, corners)
            find_corners_results = {k: cv2.findChessboardCorners(f, pattern_size, None) for k, f in
                                    gray_frames.items()}

            for cam1, cam2 in camera_pairs:
                if find_corners_results[cam1][0] and find_corners_results[cam2][0]:
                    # if find corners on both cameras, add to calibrate data
                    calibrate_data[(cam1, cam2)][0].append(objp)
                    corners1 = cv2.cornerSubPix(gray_frames[cam1], find_corners_results[cam1][1], (11, 11), (-1, -1),
                                                criteria)
                    corners2 = cv2.cornerSubPix(gray_frames[cam2], find_corners_results[cam2][1], (11, 11), (-1, -1),
                                                criteria)
                    calibrate_data[(cam1, cam2)][1].append(corners1)
                    calibrate_data[(cam1, cam2)][2].append(corners2)

                    # draw to frame
                    cv2.drawChessboardCorners(frames[cam1].value, pattern_size, corners1, find_corners_results[cam1][0])
                    cv2.drawChessboardCorners(frames[cam2].value, pattern_size, corners2, find_corners_results[cam2][0])

            # wait for 0.1 sec
            await asyncio.sleep(0.5)

        # Perform stereo calibration and calculate the fundamental matrix
        for pair in camera_pairs:
            ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
                calibrate_data[pair][0], calibrate_data[pair][1], calibrate_data[pair][2],
            )

            # save calibration data
            self.webcam_manager.set_calibrate_data(pair[0], pair[1], K1, D1, K2, D2, R, T)

        # update state
        self._app_state.is_camera_calibrated = True
        self._app_state.is_camera_calibrating = False

    def get_webcams_frames(self) -> dict[int, FrameDto]:
        """get batch of frames from each source"""
        if self._app_state.current_frames is None:
            return dict()

        return self._app_state.current_frames

    def get_mono_results(self) -> dict[int, MonoResultDto]:
        """get batch of mono result from each source"""
        if self._app_state.current_mono_results is None:
            return dict()

        return self._app_state.current_mono_results

    def stop_webcams_n_pipeline(self) -> None:
        """stop all webcams"""
        self.webcam_manager.stop_all()
        self.webcam_manager = None
        self._app_state.webcam_stared = False
        self._app_state.stared_device_indices = []

        # stop pipeline
        if self.pipe_task is not None:
            self.pipe_task.cancel()


AppManager.Singleton = AppManager()
