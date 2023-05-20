from dataclasses import dataclass
from typing import List, Optional, Protocol

from lotpose.frame_collector import FrameCollector
from lotpose.models.frameDto import FrameDto
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


class IAppManager(Protocol):
    app_state: AppState

    def start_webcams(self, device_indices: List[int]) -> None:
        ...

    def get_webcams_frames(self) -> List[FrameDto]:
        ...

    def stop_webcams(self) -> None:
        ...


class AppManager:
    """Singleton class for managing the app"""
    Singleton: IAppManager = None
    app_state: AppState
    webcam_manager: Optional[WebcamManager] = None
    current_frames: Optional[List[FrameDto]] = None

    def __init__(self):
        self.app_state = AppState()
        self.app_state.webcams_info = cv_utils.list_webcams()
        self.app_state.stared_device_indices = []

    def start_webcams(self, device_indices: List[int]) -> None:
        """init and start webcams"""

        assert self.webcam_manager is None, "Webcams already started"

        # init
        frame_collector = FrameCollector(tolerant_interval=frame_collector_tolerant_interval)
        self.webcam_manager = WebcamManager(device_indices, frame_collector, request_width, request_height)

        # start webcams
        self.webcam_manager.start_all()
        self.app_state.webcam_stared = True
        self.app_state.stared_device_indices = device_indices

    def get_webcams_frames(self) -> List[FrameDto]:
        """get batch of frames from each source"""
        return self.webcam_manager.get_frames()

    def stop_webcams(self) -> None:
        """stop all webcams"""
        self.webcam_manager.stop_all()
        self.webcam_manager = None
        self.app_state.webcam_stared = False
        self.app_state.stared_device_indices = []


AppManager.Singleton = AppManager()
