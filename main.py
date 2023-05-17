import cv2

from frame_collector import FrameCollector
from utils import cv_utils
from webcam_manager import WebcamManager


def main():
    # settings
    frame_collector_tolerant_interval = 30

    # init
    device_indices = [info.index for info in cv_utils.list_webcams()]
    frame_collector = FrameCollector(tolerant_interval=frame_collector_tolerant_interval)
    webcam_manager = WebcamManager(device_indices, frame_collector)

    # start webcams
    webcam_manager.start_all()

    # main loop
    while True:
        frames = webcam_manager.get_frames()

        cv2.imshow('Video Stream', frames[0].value)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # stop webcams
    webcam_manager.stop_all()

    # Close the display window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
