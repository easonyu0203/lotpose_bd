import multiprocessing as mp

import cv2

from lotpose.models.frameDto import FrameDto


class WebcamController:
    """act like a controller for a webcam"""
    device_index: int
    queue: mp.Queue
    reader_process: mp.Process

    def __init__(self, device_index: int):
        """
        :param device_index: the device index of the webcam
        """
        self.device_index = device_index
        self.queue = mp.Queue(maxsize=1)

    def start(self):
        """start the webcam and put the frames in the queue"""
        self.reader_process = mp.Process(target=self._read_frames)
        self.reader_process.start()

    def stop(self):
        """stop the webcam"""
        self.reader_process.terminate()
        self.reader_process.join()
        self.queue.close()
        self.queue.join_thread()

    def get_frame(self) -> FrameDto:
        """get a frame from the queue"""
        return self.queue.get()

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
            self.queue.put(FrameDto(self.device_index, frame))

        # Release the capture object
        capture.release()
