from typing import Tuple
import numpy as np
import cv2


class VideoCapture:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        assert self.cap.isOpened()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def read(self) -> Tuple[bool, np.ndarray]:
        assert self.cap.isOpened()
        return self.cap.read()
