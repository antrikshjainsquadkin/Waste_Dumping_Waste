"""Multi-Object Tracker using ByteTrack principles with distinct person and waste namespaces."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.common_types import BoundingBox, Detection, TrackedObject

logger = logging.getLogger(__name__)


class SingleCategoryTracker:
    """ByteTrack-style single-category tracker maintaining stable object tracks."""

    def __init__(
        self,
        namespace_prefix: str,
        high_thresh: float = 0.30,
        low_thresh: float = 0.10,
        match_thresh: float = 0.70,
        max_lost_frames: int = 30,
    ) -> None:
        self.prefix = namespace_prefix
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.match_thresh = float(match_thresh)
        self.max_lost_frames = int(max_lost_frames)

        self._next_id = 1
        self._active_tracks: Dict[int, TrackedObject] = {}
        self._lost_frames: Dict[int, int] = {}

    def reset(self) -> None:
        """Reset all tracking states."""
        self._next_id = 1
        self._active_tracks.clear()
        self._lost_frames.clear()

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        timestamp: float,
    ) -> List[TrackedObject]:
        """Update tracks using two-stage ByteTrack matching."""
        # 1. Split detections into high and low confidence pools
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        active_track_ids = list(self._active_tracks.keys())
        unmatched_tracks = set(active_track_ids)
        unmatched_high_dets = list(range(len(high_dets)))

        # 2. Stage 1: Match active tracks with high-confidence detections
        if active_track_ids and high_dets:
            matches, u_tracks, u_high = self._match(
                [self._active_tracks[tid].bbox for tid in active_track_ids],
                [d.bbox for d in high_dets],
                threshold=self.match_thresh,
            )
            for t_idx, d_idx in matches:
                tid = active_track_ids[t_idx]
                det = high_dets[d_idx]
                self._active_tracks[tid].update(det.bbox, det.confidence, frame_idx, timestamp)
                self._lost_frames[tid] = 0
                unmatched_tracks.discard(tid)

            unmatched_high_dets = [u_high[i] for i in range(len(u_high))]
        else:
            unmatched_tracks = set(active_track_ids)
            unmatched_high_dets = list(range(len(high_dets)))

        # 3. Stage 2: Match remaining tracks with low-confidence detections
        remaining_track_ids = list(unmatched_tracks)
        if remaining_track_ids and low_dets:
            matches_low, u_rem_tracks, _ = self._match(
                [self._active_tracks[tid].bbox for tid in remaining_track_ids],
                [d.bbox for d in low_dets],
                threshold=max(0.30, self.match_thresh - 0.25),
            )
            for t_idx, d_idx in matches_low:
                tid = remaining_track_ids[t_idx]
                det = low_dets[d_idx]
                self._active_tracks[tid].update(det.bbox, det.confidence, frame_idx, timestamp)
                self._lost_frames[tid] = 0
                unmatched_tracks.discard(tid)

        # 4. Increment lost frames for unmatched tracks and prune dead tracks
        for tid in unmatched_tracks:
            self._lost_frames[tid] = self._lost_frames.get(tid, 0) + 1

        dead_ids = [tid for tid, lost in self._lost_frames.items() if lost > self.max_lost_frames]
        for tid in dead_ids:
            self._active_tracks.pop(tid, None)
            self._lost_frames.pop(tid, None)

        # 5. Initialize new tracks for unmatched high-confidence detections
        for d_idx in unmatched_high_dets:
            det = high_dets[d_idx]
            tid = self._next_id
            self._next_id += 1
            track_obj = TrackedObject(
                track_id=f"{self.prefix}_{tid}",
                raw_id=tid,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                last_frame_idx=frame_idx,
                last_timestamp=timestamp,
                history=[(frame_idx, timestamp, det.bbox)],
            )
            self._active_tracks[tid] = track_obj
            self._lost_frames[tid] = 0

        # Return active tracks visible in current frame
        return [
            track
            for tid, track in self._active_tracks.items()
            if self._lost_frames.get(tid, 0) == 0
        ]

    def get_track_history(self, track_id: str) -> List[Tuple[int, float, BoundingBox]]:
        """Retrieve historical frame observations for a specific track ID."""
        for track in self._active_tracks.values():
            if track.track_id == track_id:
                return list(track.history)
        return []

    def _match(
        self,
        track_boxes: List[BoundingBox],
        det_boxes: List[BoundingBox],
        threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy IoU and distance-based bipartite matching."""
        if not track_boxes or not det_boxes:
            return [], list(range(len(track_boxes))), list(range(len(det_boxes)))

        cost_matrix = np.zeros((len(track_boxes), len(det_boxes)), dtype=np.float32)
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(det_boxes):
                iou = tb.iou(db)
                # Combined metric: IoU with fallback to center distance penalty
                center_dist = tb.distance_to(db)
                norm_dist = center_dist / max(tb.height, tb.width, 50.0)
                score = iou - (0.15 * min(1.0, norm_dist))
                cost_matrix[i, j] = score

        matched_tracks = set()
        matched_dets = set()
        matches = []

        # Sort candidate pairs descending by score
        indices = np.argsort(-cost_matrix, axis=None)
        for idx in indices:
            i, j = np.unravel_index(idx, cost_matrix.shape)
            if cost_matrix[i, j] < (1.0 - threshold):
                break
            if i in matched_tracks or j in matched_dets:
                continue
            matches.append((int(i), int(j)))
            matched_tracks.add(i)
            matched_dets.add(j)

        unmatched_tracks = [i for i in range(len(track_boxes)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(det_boxes)) if j not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets


class MultiObjectTrackingService:
    """Orchestrates separate ByteTrack instances for Person and Waste namespaces."""

    def __init__(
        self,
        high_thresh: float = 0.30,
        low_thresh: float = 0.10,
        match_thresh: float = 0.70,
        track_buffer: int = 30,
    ) -> None:
        self.person_tracker = SingleCategoryTracker(
            namespace_prefix="Person",
            high_thresh=high_thresh,
            low_thresh=low_thresh,
            match_thresh=match_thresh,
            max_lost_frames=track_buffer,
        )
        self.waste_tracker = SingleCategoryTracker(
            namespace_prefix="Waste",
            high_thresh=high_thresh,
            low_thresh=low_thresh,
            match_thresh=match_thresh,
            max_lost_frames=track_buffer,
        )

    def reset(self) -> None:
        """Reset trackers on new stream."""
        self.person_tracker.reset()
        self.waste_tracker.reset()

    def update(
        self,
        person_detections: List[Detection],
        waste_detections: List[Detection],
        frame_idx: int,
        timestamp: float,
    ) -> Tuple[List[TrackedObject], List[TrackedObject]]:
        """Update both trackers independently and return (tracked_persons, tracked_waste)."""
        tracked_persons = self.person_tracker.update(person_detections, frame_idx, timestamp)
        tracked_waste = self.waste_tracker.update(waste_detections, frame_idx, timestamp)
        return tracked_persons, tracked_waste

    def get_person_history(self, track_id: str) -> List[Tuple[int, float, BoundingBox]]:
        """Retrieve frame observation history for a tracked person."""
        return self.person_tracker.get_track_history(track_id)
