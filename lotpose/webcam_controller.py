import multiprocessing as mp
from typing import Tuple

import cv2

from lotpose.models.frameDto import FrameDto


class WebcamController:
    """act like a controller for a webcam"""
    device_index: int
    _queue: mp.Queue
    _reader_process: mp.Process
    width: int
    height: int

    def __init__(self, device_index: int, request_width: int, request_height: int):
        """
        :param device_index: the device index of the webcam
        """
        self.device_index = device_index
        self._queue = mp.Queue(maxsize=1)
        self.width, self.height = self._get_width_height(request_width, request_height)

    def start(self):
        """start the webcam and put the frames in the queue"""
        self._reader_process = mp.Process(target=self._read_frames)
        self._reader_process.start()

    def stop(self):
        """stop the webcam"""
        self._reader_process.terminate()
        self._reader_process.join()
        self._queue.close()
        self._queue.join_thread()

    def get_frame(self) -> FrameDto:
        """get a frame from the queue"""
        return self._queue.get()

    def _read_frames(self):
        """
        This function produces frames from a webcam and puts them in a queue.
        """
        capture = cv2.VideoCapture(self.device_index)

        while True:
            # Capture frame-by-frame
            ret, frame = capture.read()

            if not ret:
                break

            # Put the frame in the queue
            self._queue.put(FrameDto(self.device_index, frame))

        # Release the capture object
        capture.release()

    def _get_width_height(self, request_width: int, request_height: int) -> Tuple[int, int]:
        """request width and height from the webcam"""
        capture = cv2.VideoCapture(self.device_index)

        # Set the webcam resolution
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, request_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, request_height)

        # Get the actual width and height from the frame
        _, frame = capture.read()
        return frame.shape[1], frame.shape[0]
