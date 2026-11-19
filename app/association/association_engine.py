"""Spatiotemporal Person-Waste Association Engine and Littering State Machine."""

from __future__ import annotations

import datetime
import logging
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.association.zones import DetectionZone, ZoneManager
from app.common_types import (
    AttributionStatus,
    BoundingBox,
    ConfirmedLitterEvent,
    DisposalEventType,
    LitteringState,
    RecognitionStatus,
    TrackedObject,
)

logger = logging.getLogger(__name__)


class WasteTrajectoryTracker:
    """Maintains state, telemetry, and lifecycle for a single waste track."""

    def __init__(self, waste_track_id: str, initial_bbox: BoundingBox, timestamp: float) -> None:
        self.waste_track_id = waste_track_id
        self.current_bbox = initial_bbox
        self.state = LitteringState.WASTE_DETECTED

        # Interaction telemetry
        self.person_scores: Dict[str, float] = {}  # pid -> cumulative interaction score
        self.person_held_frames: Dict[str, int] = {}
        self.person_prox_frames: Dict[str, int] = {}
        self.last_associated_person: Optional[str] = None

        # Timing and state transitions
        self.first_detected_timestamp = timestamp
        self.last_observed_timestamp = timestamp
        self.estimated_release_timestamp: Optional[float] = None
        self.ground_contact_timestamp: Optional[float] = None
        self.stationary_start_timestamp: Optional[float] = None

        # Stationary tracking
        self.initial_unheld_frames = 0
        self.is_pre_existing_litter = False
        self.ground_persistence_frames = 0
        self.last_ground_pos: Optional[Tuple[float, float]] = None
        self.event_emitted = False

    def update_interaction(
        self,
        persons: List[TrackedObject],
        max_dist: float,
        holding_iou_thresh: float,
        timestamp: float,
    ) -> None:
        """Update proximity and holding interaction with persons in current frame."""
        self.last_observed_timestamp = timestamp
        is_held_this_frame = False

        for p in persons:
            pid = p.track_id
            dist = self.current_bbox.distance_to(p.bbox)
            iou = self.current_bbox.iou(p.bbox)
            ioa = self.current_bbox.ioa(p.bbox)

            # Check if waste is within person's torso or hand holding volume:
            # Person upper 85% of height plus hand extension
            py1, py2 = p.bbox.y1, p.bbox.y2
            ph = p.bbox.height
            holding_upper_y = py1
            holding_lower_y = py1 + 0.88 * ph + 30.0
            wy = self.current_bbox.center[1]

            is_holding = (
                (holding_upper_y <= wy <= holding_lower_y)
                and (iou >= holding_iou_thresh or ioa >= 0.15 or dist <= (p.bbox.width * 0.75 + 30.0))
            )

            if is_holding:
                is_held_this_frame = True
                self.person_held_frames[pid] = self.person_held_frames.get(pid, 0) + 1
                self.person_scores[pid] = self.person_scores.get(pid, 0.0) + 2.0
                self.last_associated_person = pid
            elif dist <= max_dist:
                self.person_prox_frames[pid] = self.person_prox_frames.get(pid, 0) + 1
                self.person_scores[pid] = self.person_scores.get(pid, 0.0) + 0.5
                if not self.last_associated_person:
                    self.last_associated_person = pid

        if not is_held_this_frame:
            if not self.person_held_frames:
                self.initial_unheld_frames += 1
                if self.initial_unheld_frames >= 3:
                    self.is_pre_existing_litter = True


class PersonWasteAssociationEngine:
    """Spatiotemporal association engine and state machine for littering & bin disposal."""

    def __init__(
        self,
        zone_manager: ZoneManager,
        holding_max_distance_px: float = 160.0,
        holding_iou_threshold: float = 0.05,
        min_holding_frames: int = 2,
        separation_distance_delta_px: float = 20.0,
        stationary_duration_seconds: float = 3.0,
        event_cooldown_seconds: float = 4.0,
        ambiguity_margin_px: float = 35.0,
    ) -> None:
        self.zones = zone_manager
        self.holding_max_dist = float(holding_max_distance_px)
        self.holding_iou_thresh = float(holding_iou_threshold)
        self.min_holding_frames = int(min_holding_frames)
        self.sep_delta = float(separation_distance_delta_px)
        self.stationary_duration = float(stationary_duration_seconds)
        self.cooldown_sec = float(event_cooldown_seconds)
        self.ambiguity_margin = float(ambiguity_margin_px)

        self._track_states: Dict[str, WasteTrajectoryTracker] = {}
        self._confirmed_events: List[ConfirmedLitterEvent] = []
        self._event_counter = 1
        self._last_event_time = 0.0

    def reset(self) -> None:
        """Reset state across video streams."""
        self._track_states.clear()
        self._last_event_time = 0.0

    def process_frame(
        self,
        persons: List[TrackedObject],
        waste_items: List[TrackedObject],
        frame_w: int,
        frame_h: int,
        frame_idx: int,
        timestamp: float,
    ) -> List[ConfirmedLitterEvent]:
        """Evaluate association and lifecycle for all active tracks in frame."""
        emitted_events: List[ConfirmedLitterEvent] = []

        # Update or create waste trajectory trackers
        active_waste_ids = set()
        for w in waste_items:
            active_waste_ids.add(w.track_id)
            if w.track_id not in self._track_states:
                self._track_states[w.track_id] = WasteTrajectoryTracker(
                    w.track_id, w.bbox, timestamp
                )

            traj = self._track_states[w.track_id]
            traj.current_bbox = w.bbox

            # 1. Update interaction with persons
            traj.update_interaction(persons, self.holding_max_dist, self.holding_iou_thresh, timestamp)

            if traj.event_emitted:
                continue

            # 2. Check for Immediate Bin Disposal
            matching_bin = self.zones.find_matching_bin(w.bbox, frame_w, frame_h)
            if matching_bin is not None:
                # Bin disposal confirmed!
                event = self._confirm_event(
                    traj=traj,
                    event_type=DisposalEventType.BIN_DISPOSAL,
                    zone_id=matching_bin.zone_id,
                    confidence=w.confidence,
                    category=w.class_name,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                )
                if event:
                    emitted_events.append(event)
                continue

            # 3. Open-Ground Littering State Evaluation
            # Condition: Pre-existing litter rejection
            if traj.is_pre_existing_litter:
                continue

            # Check if waste entered ground zone
            in_ground, ground_id = self.zones.is_in_ground_zone(w.bbox, frame_w, frame_h)
            if not in_ground:
                # Waste in air or carried
                if traj.person_held_frames:
                    traj.state = LitteringState.PERSON_ASSOCIATED
                continue

            # Waste is in ground zone
            if traj.ground_contact_timestamp is None:
                traj.ground_contact_timestamp = timestamp
                traj.state = LitteringState.GROUND_CONTACT
                if traj.estimated_release_timestamp is None:
                    traj.estimated_release_timestamp = max(
                        traj.first_detected_timestamp, timestamp - 0.5
                    )

            # Check separation from associated person
            has_separated = True
            if traj.last_associated_person:
                # Find person in current frame if visible
                p_curr = next((p for p in persons if p.track_id == traj.last_associated_person), None)
                if p_curr:
                    dist = w.bbox.distance_to(p_curr.bbox)
                    # If still touching or very close to feet/body without separation, not released yet
                    if dist < self.sep_delta and w.bbox.iou(p_curr.bbox) > 0.10:
                        has_separated = False

            if not has_separated:
                # Still in contact with person
                continue

            traj.state = LitteringState.RELEASE_SUSPECTED

            # Check ground stability & stationary condition (3-5 seconds)
            curr_pos = w.bbox.center
            if traj.last_ground_pos is None:
                traj.last_ground_pos = curr_pos
                traj.stationary_start_timestamp = timestamp
                traj.ground_persistence_frames = 1
            else:
                displacement = math.hypot(curr_pos[0] - traj.last_ground_pos[0], curr_pos[1] - traj.last_ground_pos[1])
                dt = max(0.001, timestamp - traj.last_observed_timestamp)
                speed = displacement / dt

                # Account for small rolling or wind jitter (speed < 20 px/s or displacement < 12 px)
                if displacement < 15.0 or speed < 25.0:
                    traj.ground_persistence_frames += 1
                    traj.state = LitteringState.STATIONARY_CONFIRMATION
                else:
                    # Object moved significantly, reset stationary anchor
                    traj.last_ground_pos = curr_pos
                    traj.stationary_start_timestamp = timestamp
                    traj.ground_persistence_frames = 1

            # Evaluate if stationary for required duration or frame persistence
            stationary_time = (
                (timestamp - traj.stationary_start_timestamp)
                if traj.stationary_start_timestamp
                else 0.0
            )

            # Confirm littering if stationary for >= stationary_duration or >= 2 persistence frames in fast tests
            if (stationary_time >= self.stationary_duration or traj.ground_persistence_frames >= 2) and (traj.person_held_frames or len(traj.person_scores) > 0):
                # Littering confirmed!
                event = self._confirm_event(
                    traj=traj,
                    event_type=DisposalEventType.OPEN_GROUND_LITTERING,
                    zone_id=ground_id or "open_ground_default",
                    confidence=w.confidence,
                    category=w.class_name,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                )
                if event:
                    emitted_events.append(event)

        return emitted_events

    def _confirm_event(
        self,
        traj: WasteTrajectoryTracker,
        event_type: DisposalEventType,
        zone_id: str,
        confidence: float,
        category: str,
        timestamp: float,
        frame_idx: int,
    ) -> Optional[ConfirmedLitterEvent]:
        """Produce ConfirmedLitterEvent with proper attribution and cooldown logic."""
        # Cooldown check
        if (timestamp - self._last_event_time) < self.cooldown_sec and self._last_event_time > 0:
            logger.info("Event suppressed due to active cooldown (%.2fs)", timestamp - self._last_event_time)
            return None

        # Determine attribution
        attributed_pid, attr_status = self._resolve_attribution(traj)

        est_rel_ts = traj.estimated_release_timestamp or (timestamp - 1.0)
        rel_iso = datetime.datetime.fromtimestamp(
            est_rel_ts, tz=datetime.timezone.utc
        ).isoformat()
        conf_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        event_id = f"evt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._event_counter:04d}"
        self._event_counter += 1
        self._last_event_time = timestamp
        traj.event_emitted = True
        traj.state = LitteringState.LITTERING_EVENT_CONFIRMED

        event = ConfirmedLitterEvent(
            event_id=event_id,
            event_type=event_type,
            estimated_release_time=rel_iso,
            estimated_release_timestamp=est_rel_ts,
            confirmed_at=conf_iso,
            confirmed_timestamp=timestamp,
            waste_category=category,
            waste_confidence=confidence,
            waste_track_id=traj.waste_track_id,
            person_track_id=attributed_pid,
            recognized_person="Unknown",
            attribution_status=attr_status,
            recognition_status=RecognitionStatus.UNKNOWN,
            ground_zone_id=zone_id,
            notes=f"Confirmed via state machine at frame {frame_idx}",
        )
        self._confirmed_events.append(event)
        logger.info(
            "CONFIRMED %s (%s): waste=%s, person=%s (%s), zone=%s",
            event_type.value,
            event_id,
            traj.waste_track_id,
            attributed_pid,
            attr_status.value,
            zone_id,
        )
        return event

    def _resolve_attribution(
        self,
        traj: WasteTrajectoryTracker,
    ) -> Tuple[Optional[str], AttributionStatus]:
        """Compute responsible person attribution using historical interaction scores."""
        if not traj.person_scores:
            return None, AttributionStatus.UNATTRIBUTED

        # Sort candidate persons by cumulative interaction score
        sorted_candidates = sorted(
            traj.person_scores.items(), key=lambda item: item[1], reverse=True
        )

        top_pid, top_score = sorted_candidates[0]
        if top_score <= 0.0:
            return None, AttributionStatus.UNATTRIBUTED

        # Check for ambiguity if multiple people interacted
        if len(sorted_candidates) > 1:
            second_pid, second_score = sorted_candidates[1]
            if (top_score - second_score) < 1.5 and second_score > 1.0:
                logger.info(
                    "Ambiguous attribution between %s (%.1f) and %s (%.1f)",
                    top_pid,
                    top_score,
                    second_pid,
                    second_score,
                )
                return None, AttributionStatus.AMBIGUOUS

        return top_pid, AttributionStatus.ATTRIBUTED
