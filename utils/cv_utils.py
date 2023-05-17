from typing import List
from dataclasses import dataclass
import cv2


@dataclass
class WebcamDeviceInfo:
    device_name: str
    index: int


def list_webcams() -> List[WebcamDeviceInfo]:
    """
    Lists all available webcams in the system.

    Returns:
        A list of WebcamDeviceInfo objects, where each object contains the device name and index of a webcam.
    """
    index = 0
    devices = []

    while True:
        # Try to capture video from the webcam
        capture = cv2.VideoCapture(index)

        if not capture.isOpened():
            break

        device_name = capture.getBackendName()

        # Release the webcam capture object
        capture.release()

        devices.append(WebcamDeviceInfo(device_name, index))

        index += 1

    return devices


if __name__ == "__main__":

    # Call the function to list all webcams
    webcams = list_webcams()

    # Print the list of webcams
    for webcam in webcams:
        print(f"Device Name: {webcam.device_name}  Index: {webcam.index}")
