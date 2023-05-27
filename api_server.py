import asyncio
import time
from typing import List

import cv2
from fastapi import FastAPI, BackgroundTasks
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


@app.get("/", response_model=None)
async def app_state():
    return AppManager.Singleton.get_app_state_dto()


@app.get("/list-webcams", response_model=List[WebcamDeviceInfo])
async def list_webcams():
    return AppManager.Singleton.get_app_state_dto().webcams_info


@app.post("/start-webcams", response_model=MsgResponse)
async def start_webcams(device_indices: List[int], background_tasks: BackgroundTasks):
    # start app
    AppManager.Singleton.start_webcams(device_indices)

    background_tasks.add_task(AppManager.Singleton.start_pipeline_bg_task)

    return MsgResponse(msg="Webcams started")


@app.post("/calibrate-camera", response_model=MsgResponse)
async def calibrate_camera(background_tasks: BackgroundTasks):
    assert AppManager.Singleton.get_app_state_dto().webcam_stared, "Webcams not started"

    background_tasks.add_task(AppManager.Singleton.start_calibration_bg_task)

    return MsgResponse(msg="Calibration started")


@app.post("/stop-webcams", response_model=MsgResponse)
async def stop_webcams():
    # stop app
    AppManager.Singleton.stop_webcams_n_pipeline()

    return MsgResponse(msg="Webcams stopped")


@app.get("/get-stream/{device_index}")
async def get_stream(device_index: int):
    # Define a generator function to retrieve video frames
    async def generate_frames():
        while True:
            if not AppManager.Singleton.get_app_state_dto().webcam_stared:
                break

            # Read the next frame from the video capture
            frames = AppManager.Singleton.get_webcams_frames()[device_index].value
            mono_results = AppManager.Singleton.get_mono_results()
            try:
                annotated_img = mono_results[device_index].annotated_img
            except KeyError:
                await asyncio.sleep(0.016)
                continue

            # Convert the frame to JPEG format
            _, jpeg = cv2.imencode('.jpg', frames)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            await asyncio.sleep(0.016)

    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.on_event("shutdown")
async def shutdown_event():
    if AppManager.Singleton.get_app_state_dto().webcam_stared:
        AppManager.Singleton.stop_webcams_n_pipeline()
    print("shutdown_event")
