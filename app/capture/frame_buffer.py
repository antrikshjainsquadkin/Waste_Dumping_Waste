"""Bounded in-memory rolling frame buffer for pre/post event evidence recording."""

from __future__ import annotations

import collections
import threading
from typing import List, Optional, Tuple
import numpy as np


class RollingFrameBuffer:
    """Thread-safe bounded ring buffer holding timestamped video frames."""

    def __init__(self, max_seconds: float = 12.0, assumed_fps: float = 30.0, min_frames: int = 10) -> None:
        self.max_seconds = float(max_seconds)
        self.max_frames = max(min_frames, int(round(self.max_seconds * assumed_fps)))
        # Stores tuples: (monotonic_timestamp, frame_bgr_copy, frame_index)
        self._buffer: collections.deque[Tuple[float, np.ndarray, int]] = collections.deque(maxlen=self.max_frames)
        self._lock = threading.Lock()

    def append(self, timestamp: float, frame: np.ndarray, frame_idx: int) -> None:
        """Append a frame to the ring buffer with defensive copying."""
        # Ensure memory efficiency: copy frame
        frame_copy = frame.copy()
        with self._lock:
            self._buffer.append((timestamp, frame_copy, frame_idx))

    def get_window(self, start_ts: float, end_ts: float) -> List[Tuple[float, np.ndarray, int]]:
        """Retrieve frames within [start_ts, end_ts] chronologically."""
        with self._lock:
            frames = [
                (ts, f.copy(), idx)
                for ts, f, idx in self._buffer
                if start_ts <= ts <= end_ts
            ]
        return frames

    def get_latest_frame(self) -> Optional[Tuple[float, np.ndarray, int]]:
        """Retrieve the most recent frame."""
        with self._lock:
            if not self._buffer:
                return None
            ts, f, idx = self._buffer[-1]
            return (ts, f.copy(), idx)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
