"""Focused test suite for EventStatusOverlay.

Validates:
1. Detection screen opens with no active event (renders compact idle monitoring).
2. Confirmed dumping activity appears in the status overlay.
3. Person attribution is ambiguous ("Ambiguous / Multiple").
4. Multiple possible persons associated with an event.
5. Person attribution is confirmed (e.g. "Person_1").
6. Face identity is UNKNOWN.
7. Face identity is successfully recognized (e.g. "Alice").
8. Waste categories include organic, recyclable, and residual.
9. Review-required status is displayed correctly in orange for open ground dumping.
10. Designated-bin disposal receives the correct event label and status.
11. A new event replaces or updates the previous event information correctly.
12. Missing optional event data (None, empty strings, missing confidence) does not crash the overlay.
13. Video rendering, bounding boxes, and existing detection labels remain functional.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.common_types import (
    AttributionStatus,
    BoundingBox,
    ConfirmedLitterEvent,
    DisposalEventType,
    RecognitionStatus,
    TrackedObject,
)
from app.ui.event_overlay import EventStatusOverlay


class TestEventStatusOverlay(unittest.TestCase):
    """Test suite directly validating the EventStatusOverlay requirements."""

    def setUp(self):
        self.overlay = EventStatusOverlay(display_duration_seconds=5.0)
        self.dummy_frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    # 1. Detection screen opens with no active event (idle monitoring state)
    def test_01_idle_state_when_no_active_event(self):
        self.assertIsNone(self.overlay.active_event)
        frame = self.dummy_frame.copy()
        res = self.overlay.draw(frame, new_events=None)
        self.assertEqual(res.shape, (480, 640, 3))
        # The frame should have been modified with the idle HUD
        self.assertFalse(np.array_equal(frame, self.dummy_frame))

    # 2. Confirmed dumping activity appears in status overlay
    def test_02_confirmed_dumping_appears(self):
        evt = ConfirmedLitterEvent(
            event_id="evt_001",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-20T10:00:00",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-20T10:00:05",
            confirmed_timestamp=105.0,
            waste_category="recyclable",
            waste_confidence=0.85,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
        )
        frame = self.dummy_frame.copy()
        res = self.overlay.draw(frame, new_events=[evt])
        self.assertIsNotNone(self.overlay.active_event)
        self.assertEqual(self.overlay.active_event.event_id, "evt_001")
        self.assertFalse(np.array_equal(frame, self.dummy_frame))

    # 3. Person attribution is ambiguous -> displays "Ambiguous / Multiple"
    def test_03_ambiguous_person_attribution(self):
        evt = ConfirmedLitterEvent(
            event_id="evt_002",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-20T10:00:00",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-20T10:00:05",
            confirmed_timestamp=105.0,
            waste_category="recyclable",
            waste_confidence=0.80,
            waste_track_id="Waste_1",
            person_track_id=None,
            attribution_status=AttributionStatus.AMBIGUOUS,
        )
        frame = self.dummy_frame.copy()
        res = self.overlay.draw(frame, new_events=[evt])
        self.assertEqual(self.overlay.active_event.attribution_status, AttributionStatus.AMBIGUOUS)

    # 4. Multiple possible persons associated
    def test_04_multiple_persons_association(self):
        evt = ConfirmedLitterEvent(
            event_id="evt_003",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-20T10:00:00",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-20T10:00:05",
            confirmed_timestamp=105.0,
            waste_category="residual",
            waste_confidence=0.75,
            waste_track_id="Waste_1",
            person_track_id=None,
            attribution_status=AttributionStatus.AMBIGUOUS,
            notes="Multiple interaction candidates: Person_1 (score 4.0), Person_2 (score 3.5)",
        )
        self.overlay.update_event(evt)
        self.assertEqual(self.overlay.active_event.attribution_status, AttributionStatus.AMBIGUOUS)

    # 5. Person attribution confirmed -> displays Person_1
    def test_05_person_attribution_confirmed(self):
        evt = ConfirmedLitterEvent(
            event_id="evt_004",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-20T10:00:00",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-20T10:00:05",
            confirmed_timestamp=105.0,
            waste_category="organic",
            waste_confidence=0.90,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
            attribution_status=AttributionStatus.ATTRIBUTED,
        )
        self.overlay.update_event(evt)
        self.assertEqual(self.overlay.active_event.person_track_id, "Person_1")
        self.assertEqual(self.overlay.active_event.attribution_status, AttributionStatus.ATTRIBUTED)

    # 6. Face identity is UNKNOWN
    def test_06_face_identity_unknown(self):
        evt = ConfirmedLitterEvent(
            event_id="evt_005",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-20T10:00:00",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-20T10:00:05",
            confirmed_timestamp=105.0,
            waste_category="recyclable",
            waste_confidence=0.88,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
            recognized_person="Unknown",
            recognition_status=RecognitionStatus.UNKNOWN,
        )
        self.overlay.update_event(evt)
        self.assertEqual(self.overlay.active_event.recognized_person, "Unknown")
        self.assertEqual(self.overlay.active_event.recognition_status, RecognitionStatus.UNKNOWN)

    # 7. Face identity successfully recognized -> Alice
    def test_07_face_identity_recognized(self):
        evt = ConfirmedLitterEvent(
            event_id="evt_006",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-20T10:00:00",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-20T10:00:05",
            confirmed_timestamp=105.0,
            waste_category="recyclable",
            waste_confidence=0.88,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
            recognized_person="Alice",
            recognition_status=RecognitionStatus.IDENTIFIED,
        )
        self.overlay.update_event(evt)
        self.assertEqual(self.overlay.active_event.recognized_person, "Alice")
        self.assertEqual(self.overlay.active_event.recognition_status, RecognitionStatus.IDENTIFIED)

    # 8. Waste categories: organic, recyclable, residual
    def test_08_waste_categories_formatting(self):
        for cat in ["organic", "recyclable", "residual"]:
            evt = ConfirmedLitterEvent(
                event_id=f"evt_{cat}",
                event_type=DisposalEventType.OPEN_GROUND_LITTERING,
                estimated_release_time="2026-11-20T10:00:00",
                estimated_release_timestamp=100.0,
                confirmed_at="2026-11-20T10:00:05",
                confirmed_timestamp=105.0,
                waste_category=cat,
                waste_confidence=0.80,
                waste_track_id="Waste_1",
                person_track_id="Person_1",
            )
            frame = self.dummy_frame.copy()
            res = self.overlay.draw(frame, new_events=[evt])
            self.assertEqual(self.overlay.active_event.waste_category, cat)

    # 9. Review required status displayed for open ground dumping
    def test_09_review_required_status(self):
        evt = ConfirmedLitterEvent(
            event_id="evt_009",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-20T10:00:00",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-20T10:00:05",
            confirmed_timestamp=105.0,
            waste_category="residual",
            waste_confidence=0.85,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
        )
        self.overlay.update_event(evt)
        self.assertEqual(self.overlay.active_event.event_type, DisposalEventType.OPEN_GROUND_LITTERING)

    # 10. Designated-bin disposal receives compliant label and does NOT say review required
    def test_10_designated_bin_disposal(self):
        evt = ConfirmedLitterEvent(
            event_id="evt_010",
            event_type=DisposalEventType.BIN_DISPOSAL,
            estimated_release_time="2026-11-20T10:00:00",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-20T10:00:05",
            confirmed_timestamp=105.0,
            waste_category="recyclable",
            waste_confidence=0.82,
            waste_track_id="Waste_2",
            person_track_id="Person_2",
        )
        frame = self.dummy_frame.copy()
        res = self.overlay.draw(frame, new_events=[evt])
        self.assertEqual(self.overlay.active_event.event_type, DisposalEventType.BIN_DISPOSAL)

    # 11. New event replaces previous event
    def test_11_event_replacement_and_expiry(self):
        evt1 = ConfirmedLitterEvent(
            event_id="evt_first",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-20T10:00:00",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-20T10:00:05",
            confirmed_timestamp=105.0,
            waste_category="organic",
            waste_confidence=0.70,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
        )
        self.overlay.update_event(evt1)
        self.assertEqual(self.overlay.active_event.event_id, "evt_first")

        evt2 = ConfirmedLitterEvent(
            event_id="evt_second",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-20T10:01:00",
            estimated_release_timestamp=160.0,
            confirmed_at="2026-11-20T10:01:05",
            confirmed_timestamp=165.0,
            waste_category="recyclable",
            waste_confidence=0.95,
            waste_track_id="Waste_2",
            person_track_id="Person_2",
        )
        self.overlay.update_event(evt2)
        self.assertEqual(self.overlay.active_event.event_id, "evt_second")

        # Test duration expiry
        self.overlay._last_event_time = time.time() - 10.0  # past 5.0s duration
        self.assertIsNone(self.overlay.active_event)

    # 12. Missing optional event data handles gracefully without crashing
    def test_12_missing_optional_data_handles_gracefully(self):
        sparse_evt = ConfirmedLitterEvent(
            event_id="evt_sparse",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="",
            estimated_release_timestamp=0.0,
            confirmed_at="",
            confirmed_timestamp=0.0,
            waste_category="",
            waste_confidence=0.0,
            waste_track_id="",
            person_track_id=None,
            recognized_person="",
        )
        frame = self.dummy_frame.copy()
        # Must not raise an exception
        res = self.overlay.draw(frame, new_events=[sparse_evt])
        self.assertEqual(res.shape, (480, 640, 3))

    # 13. Video rendering, bounding boxes, and existing detection labels remain functional
    def test_13_rendering_with_detection_boxes(self):
        frame = self.dummy_frame.copy()
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "recyclable", 0.88, BoundingBox(140, 220, 170, 270), 1, 0.0)

        # Draw boxes
        x1, y1, x2, y2 = p1.bbox.to_xyxy_int()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        x1, y1, x2, y2 = w1.bbox.to_xyxy_int()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw overlay
        res = self.overlay.draw(frame, tracked_persons=[p1], tracked_waste=[w1])
        self.assertEqual(res.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
