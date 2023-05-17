from typing import List

import cv2
import uvicorn as uvicorn
from fastapi import FastAPI

from lotpose.frame_collector import FrameCollector
from utils import cv_utils
from lotpose.webcam_manager import WebcamManager
from utils.cv_utils import WebcamDeviceInfo

app = FastAPI(
    arbitrary_types_allowed=True
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/list-webcams", response_model=List[WebcamDeviceInfo])
async def list_webcams():
    return cv_utils.list_webcams()


def main():
    # settings
    frame_collector_tolerant_interval = 30
    request_width = 640
    request_height = 480

    # init
    device_indices = [info.index for info in cv_utils.list_webcams()]
    frame_collector = FrameCollector(tolerant_interval=frame_collector_tolerant_interval)
    webcam_manager = WebcamManager(device_indices, frame_collector, request_width, request_height)

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
