import cv2
from typing import Generator
import numpy as np


class VideoStream:
    def __init__(self, source: int | str = 0):
        self.source = source
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def frames(self) -> Generator[np.ndarray, None, None]:
        if not self.open():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        try:
            while True:
                ret, frame = self.read()
                if not ret:
                    break
                yield frame
        finally:
            self.release()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def width(self) -> int:
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return 0

    @property
    def height(self) -> int:
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return 0

    @property
    def fps(self) -> float:
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0

