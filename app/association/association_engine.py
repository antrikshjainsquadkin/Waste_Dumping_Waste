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

        # Stationary & motion telemetry
        self.initial_unheld_frames = 0
        self.is_pre_existing_litter = False
        self.ground_persistence_frames = 0
        self.prev_center: Optional[Tuple[float, float]] = None
        self.prev_timestamp: Optional[float] = None
        self.stationary_anchor_pos: Optional[Tuple[float, float]] = None
        self.is_rolling = False
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

        # If waste is already established as stationary on the ground, a person standing
        # or walking near it without moving it is NOT holding it.
        is_stationary_on_ground = (
            self.stationary_anchor_pos is not None
            and not self.is_rolling
            and math.hypot(
                self.current_bbox.center[0] - self.stationary_anchor_pos[0],
                self.current_bbox.center[1] - self.stationary_anchor_pos[1]
            ) < 25.0
        )

        for p in persons:
            pid = p.track_id
            dist = self.current_bbox.distance_to(p.bbox)
            iou = self.current_bbox.iou(p.bbox)
            ioa = self.current_bbox.ioa(p.bbox)

            # Check if waste is within person's torso or hand holding volume:
            # Below cranial head zone (> 15% height) and excluding ground-level foot zone (< 85% height)
            py1, py2 = p.bbox.y1, p.bbox.y2
            ph = p.bbox.height
            holding_upper_y = py1 + 0.15 * ph
            holding_lower_y = py1 + 0.85 * ph
            wy = self.current_bbox.center[1]

            is_in_hand_volume = (holding_upper_y <= wy <= holding_lower_y)

            # Spatial overlap requires genuine physical intersection or touching contact.
            # Center-to-center distance alone must NEVER qualify as holding.
            dx = max(0.0, max(p.bbox.x1 - self.current_bbox.x2, self.current_bbox.x1 - p.bbox.x2))
            dy = max(0.0, max(p.bbox.y1 - self.current_bbox.y2, self.current_bbox.y1 - p.bbox.y2))
            box_separation = math.hypot(dx, dy)

            is_spatial_overlap = (
                iou >= holding_iou_thresh
                or ioa >= 0.10
                or box_separation <= 10.0
            )

            is_holding = is_in_hand_volume and is_spatial_overlap and (not is_stationary_on_ground)

            if is_holding:
                is_held_this_frame = True
                self.person_held_frames[pid] = self.person_held_frames.get(pid, 0) + 1
                self.person_scores[pid] = self.person_scores.get(pid, 0.0) + 3.0
                self.last_associated_person = pid
            elif dist <= max_dist:
                self.person_prox_frames[pid] = self.person_prox_frames.get(pid, 0) + 1
                self.person_scores[pid] = self.person_scores.get(pid, 0.0) + 0.5
                # Proximity alone does NOT set last_associated_person

        if not is_held_this_frame:
            if not self.person_held_frames:
                self.initial_unheld_frames += 1
                if self.initial_unheld_frames >= 2:
                    self.is_pre_existing_litter = True
                if self.stationary_anchor_pos is None:
                    self.stationary_anchor_pos = self.current_bbox.center


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

    def get_trajectory(self, track_id: str) -> Optional[WasteTrajectoryTracker]:
        """Retrieve tracking state for a given waste track ID."""
        return self._track_states.get(track_id)

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
            # Condition 1: Pre-existing litter rejection
            if traj.is_pre_existing_litter:
                continue

            # Condition 2: Genuine open-ground dumping requires that a person actually held or carried
            # the waste prior to ground contact. If the waste was never held by any person,
            # it is classified as pre-existing litter and does not emit a dumping event.
            if not traj.person_held_frames:
                if traj.initial_unheld_frames >= 2 or traj.ground_contact_timestamp is not None:
                    traj.is_pre_existing_litter = True
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
                    py1, py2 = p_curr.bbox.y1, p_curr.bbox.y2
                    ph = p_curr.bbox.height
                    wy = w.bbox.center[1]
                    # If still inside person's holding volume or overlapping, not released yet
                    if (py1 <= wy <= py1 + 0.85 * ph) and (
                        dist < self.holding_max_dist * 0.65
                        or w.bbox.iou(p_curr.bbox) > 0.03
                        or w.bbox.ioa(p_curr.bbox) >= 0.10
                    ):
                        has_separated = False

            if not has_separated:
                # Still in contact with person
                continue

            traj.state = LitteringState.RELEASE_SUSPECTED

            # Check ground stability & stationary condition (3-5 seconds)
            curr_pos = w.bbox.center

            # Calculate frame-to-frame displacement and instant speed
            if traj.prev_timestamp is not None and timestamp > traj.prev_timestamp:
                dt = timestamp - traj.prev_timestamp
                step_disp = math.hypot(curr_pos[0] - traj.prev_center[0], curr_pos[1] - traj.prev_center[1])
                speed = step_disp / max(0.001, dt)
            else:
                speed = 0.0
                step_disp = 0.0

            traj.prev_center = curr_pos
            traj.prev_timestamp = timestamp

            # Anchor displacement tracking
            if traj.stationary_anchor_pos is None:
                traj.stationary_anchor_pos = curr_pos
                traj.stationary_start_timestamp = timestamp
                traj.ground_persistence_frames = 1
                traj.is_rolling = False
                anchor_disp = 0.0
            else:
                anchor_disp = math.hypot(
                    curr_pos[0] - traj.stationary_anchor_pos[0],
                    curr_pos[1] - traj.stationary_anchor_pos[1],
                )

            # Object is stationary ONLY if BOTH instant speed is low (<25 px/s) AND anchor displacement is small (<25 px)
            if speed < 25.0 and anchor_disp < 25.0:
                traj.ground_persistence_frames += 1
                traj.state = LitteringState.STATIONARY_CONFIRMATION
                traj.is_rolling = False
            else:
                # Waste moves or rolls after impact! Reset stationary anchor and timer
                traj.stationary_anchor_pos = curr_pos
                traj.stationary_start_timestamp = timestamp
                traj.ground_persistence_frames = 1
                traj.is_rolling = True
                traj.state = LitteringState.GROUND_CONTACT
                logger.debug(
                    "Waste %s in motion/rolling: speed=%.1f px/s, anchor_disp=%.1f px (timer reset)",
                    traj.waste_track_id, speed, anchor_disp
                )

            # Evaluate if stationary for required duration (configurable 3.0 to 5.0 seconds)
            stationary_time = (
                (timestamp - traj.stationary_start_timestamp)
                if traj.stationary_start_timestamp is not None
                else 0.0
            )

            # Must satisfy full stationary_duration without rolling
            if stationary_time >= self.stationary_duration and not traj.is_rolling:
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
        # Convert monotonic offset to true wall-clock UTC timestamp
        wall_now = datetime.datetime.now(datetime.timezone.utc)
        elapsed_since_rel = max(0.0, timestamp - est_rel_ts)
        rel_time = wall_now - datetime.timedelta(seconds=elapsed_since_rel)
        rel_iso = rel_time.isoformat()
        conf_iso = wall_now.isoformat()

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
        """Compute responsible person attribution using verified holding and trajectory evidence."""
        # Genuine attribution requires that the person actually held/carried the waste
        # Bystanders who merely walked or stood nearby (0 held frames) are strictly filtered out.
        held_candidates = {
            pid: score for pid, score in traj.person_scores.items()
            if traj.person_held_frames.get(pid, 0) >= 1
        }

        if not held_candidates:
            # No person ever held or carried the waste
            return None, AttributionStatus.UNATTRIBUTED

        # Sort verified thrower candidates by cumulative interaction score
        sorted_candidates = sorted(
            held_candidates.items(), key=lambda item: item[1], reverse=True
        )

        top_pid, top_score = sorted_candidates[0]
        if top_score <= 0.0:
            return None, AttributionStatus.UNATTRIBUTED

        # Check for ambiguity if multiple people actively held/interacted with the waste
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
