"""Shared Camera Capture Service providing a single source for detection and recording."""

from __future__ import annotations

import logging
import os
import time
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

from app.capture.frame_buffer import RollingFrameBuffer

logger = logging.getLogger(__name__)


class SharedCameraService:
    """Provides video stream capture with an integrated rolling buffer."""

    def __init__(
        self,
        source: str | int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        buffer_duration_seconds: float = 12.0,
    ) -> None:
        self.source = source
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.buffer = RollingFrameBuffer(max_seconds=buffer_duration_seconds, assumed_fps=fps)

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
        self._frame_count = 0
        self._is_video_file = False
        self._fps_actual = float(fps)

    def open(self) -> bool:
        """Open camera stream or video file."""
        src = self.source
        if isinstance(src, str):
            if src.isdigit():
                src = int(src)
            elif os.path.isfile(src):
                self._is_video_file = True

        logger.info("Opening SharedCameraService with source: %s", src)
        if isinstance(src, int) and os.name == "nt":
            self._cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(src)
        else:
            self._cap = cv2.VideoCapture(src)

        if not self._cap or not self._cap.isOpened():
            logger.error("Failed to open video source: %s", self.source)
            self._is_opened = False
            return False

        if not self._is_video_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        else:
            file_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if file_fps and file_fps > 0:
                self._fps_actual = file_fps

        self._is_opened = True
        self._frame_count = 0
        logger.info("SharedCameraService opened successfully (FPS: %.2f)", self._fps_actual)
        return True

    @property
    def fps(self) -> float:
        return self._fps_actual

    @property
    def is_opened(self) -> bool:
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float, int]:
        """Read single frame, append to buffer, return (success, frame, timestamp, frame_idx)."""
        if not self.is_opened or self._cap is None:
            return False, None, 0.0, -1

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            return False, None, 0.0, -1

        self._frame_count += 1
        ts = time.monotonic()
        self.buffer.append(ts, frame, self._frame_count)
        return True, frame, ts, self._frame_count

    def stream(self) -> Generator[Tuple[np.ndarray, float, int], None, None]:
        """Generator yielding (frame, timestamp, frame_idx) until stream ends."""
        while self.is_opened:
            ret, frame, ts, idx = self.read_frame()
            if not ret or frame is None:
                break
            yield frame, ts, idx

    def close(self) -> None:
        """Release capture device and clean up buffer."""
        if self._cap:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing cv2.VideoCapture: %s", e)
            self._cap = None
        self._is_opened = False
        logger.info("SharedCameraService closed.")
