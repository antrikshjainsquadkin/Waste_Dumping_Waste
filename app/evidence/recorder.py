"""Evidence Recording Service saving 5s-before/5s-after video clips and face snapshots."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.capture.frame_buffer import RollingFrameBuffer
from app.common_types import ConfirmedLitterEvent

logger = logging.getLogger(__name__)


class EvidenceRecorder:
    """Saves video evidence clips and face snapshot images."""

    def __init__(
        self,
        evidence_dir: str | Path,
        frame_buffer: RollingFrameBuffer,
        pre_event_seconds: float = 5.0,
        post_event_seconds: float = 5.0,
        codec: str = "mp4v",
    ) -> None:
        self.evidence_dir = Path(evidence_dir)
        self.frame_buffer = frame_buffer
        self.pre_event_seconds = float(pre_event_seconds)
        self.post_event_seconds = float(post_event_seconds)
        self.codec = str(codec)

        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        # Pending events waiting for post-event frames to accumulate: (event, flush_timestamp, face_crop)
        self._pending_clips: List[Tuple[ConfirmedLitterEvent, float, Optional[np.ndarray]]] = []

    def schedule_clip(
        self,
        event: ConfirmedLitterEvent,
        face_crop: Optional[np.ndarray] = None,
    ) -> None:
        """Schedule video clip saving, waiting for post-event seconds to elapse."""
        flush_ts = event.estimated_release_timestamp + self.post_event_seconds
        self._pending_clips.append((event, flush_ts, face_crop))
        logger.info("Scheduled evidence capture for %s (target flush in %.1fs)", event.event_id, flush_ts - event.confirmed_timestamp)

    def check_pending(self, current_timestamp: float) -> List[ConfirmedLitterEvent]:
        """Check and write any pending clips whose post-event window has elapsed."""
        ready_events: List[ConfirmedLitterEvent] = []
        still_pending = []

        for event, flush_ts, face_crop in self._pending_clips:
            if current_timestamp >= flush_ts:
                self._save_evidence(event, face_crop)
                ready_events.append(event)
            else:
                still_pending.append((event, flush_ts, face_crop))

        self._pending_clips = still_pending
        return ready_events

    def flush_all(self) -> List[ConfirmedLitterEvent]:
        """Force flush all remaining pending clips (e.g. at end of stream)."""
        flushed: List[ConfirmedLitterEvent] = []
        for event, _, face_crop in self._pending_clips:
            self._save_evidence(event, face_crop)
            flushed.append(event)
        self._pending_clips.clear()
        return flushed

    def _save_evidence(
        self,
        event: ConfirmedLitterEvent,
        face_crop: Optional[np.ndarray],
    ) -> None:
        """Write video clip and face image to local storage."""
        start_ts = event.estimated_release_timestamp - self.pre_event_seconds
        end_ts = event.estimated_release_timestamp + self.post_event_seconds

        frames = self.frame_buffer.get_window(start_ts, end_ts)
        if not frames:
            logger.warning("No frames found in rolling buffer for event %s", event.event_id)
            return

        # 1. Save video clip
        video_filename = f"{event.event_id}_evidence.mp4"
        video_path = self.evidence_dir / video_filename
        h, w = frames[0][1].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
        if writer.isOpened():
            for _, frame, _ in frames:
                writer.write(frame)
            writer.release()
            event.video_evidence_path = str(video_path)
            logger.info("Saved video evidence (%d frames) to %s", len(frames), video_path)
        else:
            logger.error("Failed to open VideoWriter for %s", video_path)

        # 2. Save face snapshot if available
        if face_crop is not None and face_crop.size > 0:
            face_filename = f"{event.event_id}_face.jpg"
            face_path = self.evidence_dir / face_filename
            cv2.imwrite(str(face_path), face_crop)
            event.face_snapshot_path = str(face_path)
            logger.info("Saved face snapshot to %s", face_path)
