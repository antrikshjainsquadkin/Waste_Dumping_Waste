"""Visual verification script for EventStatusOverlay."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.common_types import (
    AttributionStatus,
    ConfirmedLitterEvent,
    DisposalEventType,
    RecognitionStatus,
    TrackedObject,
    BoundingBox,
)
from app.ui.event_overlay import EventStatusOverlay


def generate_visual_samples(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = EventStatusOverlay(display_duration_seconds=12.0)

    # 1. Base test image (640x480 hallway/outdoor simulation)
    # Create realistic gradient background with mock person and waste bounding boxes
    base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        base_frame[y, :] = [int(130 - y * 0.15), int(140 - y * 0.12), int(155 - y * 0.1)]

    # Mock person bounding box on right side
    cv2.rectangle(base_frame, (380, 100), (520, 440), (255, 0, 0), 2)
    cv2.putText(base_frame, "Person_1", (380, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Mock waste bounding box on ground
    cv2.rectangle(base_frame, (420, 410), (460, 450), (0, 0, 255), 2)
    cv2.putText(base_frame, "Waste_1 (recyclable)", (420, 402), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Case 1: Reference Match - Ambiguous attribution, Recyclable, Review Required, Unknown identity
    evt_ambiguous = ConfirmedLitterEvent(
        event_id="evt_20261120_100101_0001",
        event_type=DisposalEventType.OPEN_GROUND_LITTERING,
        estimated_release_time="2026-11-20T10:00:55",
        estimated_release_timestamp=1000.0,
        confirmed_at="2026-11-20T10:01:01",
        confirmed_timestamp=1006.0,
        waste_category="recyclable",
        waste_confidence=0.88,
        waste_track_id="Waste_1",
        person_track_id=None,
        recognized_person="Unknown",
        attribution_status=AttributionStatus.AMBIGUOUS,
        recognition_status=RecognitionStatus.UNKNOWN,
    )

    frame_case1 = base_frame.copy()
    overlay.draw(frame_case1, new_events=[evt_ambiguous])
    path_case1 = output_dir / "overlay_case1_ambiguous_dumping.jpg"
    cv2.imwrite(str(path_case1), frame_case1)
    print(f"Saved: {path_case1}")

    # Case 2: Confirmed Attribution with Identified Face (Alice)
    evt_identified = ConfirmedLitterEvent(
        event_id="evt_20261120_100202_0002",
        event_type=DisposalEventType.OPEN_GROUND_LITTERING,
        estimated_release_time="2026-11-20T10:01:55",
        estimated_release_timestamp=1060.0,
        confirmed_at="2026-11-20T10:02:02",
        confirmed_timestamp=1067.0,
        waste_category="organic",
        waste_confidence=0.92,
        waste_track_id="Waste_2",
        person_track_id="Person_1",
        recognized_person="Alice",
        attribution_status=AttributionStatus.ATTRIBUTED,
        recognition_status=RecognitionStatus.IDENTIFIED,
    )

    frame_case2 = base_frame.copy()
    overlay.draw(frame_case2, new_events=[evt_identified])
    path_case2 = output_dir / "overlay_case2_identified_alice.jpg"
    cv2.imwrite(str(path_case2), frame_case2)
    print(f"Saved: {path_case2}")

    # Case 3: Bin Disposal (Non-offense, compliant)
    evt_bin = ConfirmedLitterEvent(
        event_id="evt_20261120_100303_0003",
        event_type=DisposalEventType.BIN_DISPOSAL,
        estimated_release_time="2026-11-20T10:02:55",
        estimated_release_timestamp=1120.0,
        confirmed_at="2026-11-20T10:03:03",
        confirmed_timestamp=1128.0,
        waste_category="residual",
        waste_confidence=0.76,
        waste_track_id="Waste_3",
        person_track_id="Person_2",
        recognized_person="Unknown",
        attribution_status=AttributionStatus.ATTRIBUTED,
        recognition_status=RecognitionStatus.UNKNOWN,
    )

    frame_case3 = base_frame.copy()
    overlay.draw(frame_case3, new_events=[evt_bin])
    path_case3 = output_dir / "overlay_case3_bin_disposal.jpg"
    cv2.imwrite(str(path_case3), frame_case3)
    print(f"Saved: {path_case3}")

    # Case 4: Idle Monitoring State (No active event)
    overlay.clear()
    dummy_persons = [TrackedObject("Person_1", 1, "person", 0.9, BoundingBox(380, 100, 520, 440), 1, 0.0)]
    dummy_waste = [TrackedObject("Waste_1", 1, "recyclable", 0.88, BoundingBox(420, 410, 460, 450), 1, 0.0)]

    frame_case4 = base_frame.copy()
    overlay.draw(frame_case4, new_events=None, tracked_persons=dummy_persons, tracked_waste=dummy_waste)
    path_case4 = output_dir / "overlay_case4_idle_monitoring.jpg"
    cv2.imwrite(str(path_case4), frame_case4)
    print(f"Saved: {path_case4}")


if __name__ == "__main__":
    out_dir = Path("reports/visual_verification")
    generate_visual_samples(out_dir)
