import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

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


@dataclass
class AppStateDto:
    webcam_stared: bool = False
    stared_device_indices: List[int] = None
    webcams_info: List[cv_utils.WebcamDeviceInfo] = None


class IAppManager(Protocol):
    _app_state: AppState

    def get_app_state_dto(self) -> AppStateDto:
        ...

    def start_webcams_n_pipeline(self, device_indices: List[int]) -> None:
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
            webcams_info=self._app_state.webcams_info
        )
        return dto

    def start_webcams_n_pipeline(self, device_indices: List[int]) -> None:
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
            # mono_results = await self.mono_landmarker.process_async(frames)

            self._app_state.current_frames = frames
            # self.app_state.current_mono_results = mono_results

            await asyncio.sleep(0.016)  # run 60 fps

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
