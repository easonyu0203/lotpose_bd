import asyncio
import time
from typing import List

import cv2
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from lotpose.frame_collector import FrameCollector
from app_manager import AppManager, AppState
from models import MsgResponse
from utils import cv_utils
from lotpose.webcam_manager import WebcamManager
from utils.cv_utils import WebcamDeviceInfo

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=AppState)
async def app_state():
    return AppManager.Singleton.app_state


@app.get("/list-webcams", response_model=List[WebcamDeviceInfo])
async def list_webcams():
    return AppManager.Singleton.app_state.webcams_info


@app.post("/start-webcams", response_model=MsgResponse)
async def start_webcams(device_indices: List[int]):
    # start app
    AppManager.Singleton.start_webcams(device_indices)

    return MsgResponse(msg="Webcams started")


@app.post("/stop-webcams", response_model=MsgResponse)
async def stop_webcams():
    # stop app
    AppManager.Singleton.stop_webcams()

    return MsgResponse(msg="Webcams stopped")


@app.get("/get-stream/{device_index}")
async def get_stream(device_index: int):
    # Define a generator function to retrieve video frames
    async def generate_frames():
        while True:
            # Read the next frame from the video capture
            frames = AppManager.Singleton.get_webcams_frames()
            frame = [frame for frame in frames if frame.device_index == device_index][0]

            # Convert the frame to JPEG format
            _, jpeg = cv2.imencode('.jpg', frame.value)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            await asyncio.sleep(0.016)

    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

