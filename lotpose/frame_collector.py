import time
from typing import Protocol, List

from lotpose.dtos.frame_dto import FrameDto


class FrameSource(Protocol):
    """Produces frames from a source"""

    device_index: int

    def get_frame(self) -> FrameDto:
        ...


class FrameCollector:
    """Collects frames from difference source and synchronizes them"""

    #
    tolerant_interval: int  # (ms)
    timeout: int
    frame_rate: int
    _obsolete_threshold_time: float
    _current_frames: dict[int, FrameDto]

    def __init__(self, tolerant_interval=30, timeout=2, frame_rate=60):
        """
        :param tolerant_interval: for a given batch of frames, the max interval between the first and the last frame(ms)
        :param timeout: maximum duration for the loop (in seconds)
        :param frame_rate: desired frame rate (in seconds)
        """
        self.tolerant_interval = tolerant_interval
        self.timeout = timeout
        self.frame_rate = frame_rate
        self._obsolete_threshold_time = 0
        self._current_frames = dict()

    def get_frames(self, frame_sources: dict[int, FrameSource]) -> dict[int, FrameDto]:
        """
        Fetches a batch of frames from each of the frame sources.

        we ensure the output frames timestamp is within tolerant interval

        :param frame_sources: list of source to collect frames from
        :return: a batch of frames
        """

        assert len(frame_sources) > 0, "frame_producers must not be empty"

        # if the current frames are not obsolete, return them
        if time.time() < self._obsolete_threshold_time:
            return self._current_frames

        frames = {device_idx: frame_src.get_frame() for device_idx, frame_src in frame_sources.items()}

        if len(frames) == 1:
            self._obsolete_threshold_time = time.time() + 1 / self.frame_rate
            self._current_frames = frames
            return frames

        start_time = time.time()  # Record the starting time

        frames_sorted = list(frames.values())
        while True:
            # sort device frames by timestamp
            frames_sorted.sort(key=lambda f: f.timestamp)

            # check is in tolerant interval
            time_diff = (frames_sorted[-1].timestamp - frames_sorted[0].timestamp)

            if time_diff <= self.tolerant_interval:
                self._obsolete_threshold_time = time.time() + 1 / self.frame_rate
                self._current_frames = frames
                return frames

            # Check if the timeout duration has been exceeded
            if time.time() - start_time > self.timeout:
                raise TimeoutError("The loop has exceeded the maximum allowed duration.")

            # renew the oldest frame
            oldest_frame = frames_sorted.pop(0)
            frames_sorted.append(frame_sources[oldest_frame.device_index].get_frame())
