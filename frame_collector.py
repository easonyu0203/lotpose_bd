from datetime import datetime
from typing import Protocol, List

from models.frameDto import FrameDto


class FrameSource(Protocol):
    """Produces frames from a source"""

    device_index: int

    def get_frame(self) -> FrameDto:
        ...


class FrameCollector:
    """Collects frames from difference source and synchronizes them"""

    #
    tolerant_interval: int  # (ms)

    def __init__(self, tolerant_interval=30):
        """
        :param tolerant_interval: for a given batch of frames, the max interval between the first and the last frame(ms)
        """
        self.tolerant_interval = tolerant_interval

    def get_frames(self, frame_sources: List[FrameSource]) -> List[FrameDto]:
        """
        Fetches a batch of frames from each of the frame sources.

        we ensure the output frames timestamp is within tolerant interval

        :param frame_sources: list of source to collect frames from
        :return: a batch of frames
        """

        assert len(frame_sources) > 0, "frame_producers must not be empty"

        frames = [frame_src.get_frame() for frame_src in frame_sources]

        if len(frames) == 1:
            return frames

        while True:
            # sort device frames by timestamp
            frames.sort(key=lambda frame: frame.timestamp)

            # check is in tolerant interval
            time_diff = (frames[-1].timestamp - frames[0].timestamp).total_seconds() * 1000

            if time_diff <= self.tolerant_interval:
                return frames

            # renew the oldest frame
            oldest_frame = frames.pop(0)
            frames.append(frame_sources[oldest_frame.device_index].get_frame())

