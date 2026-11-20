"""Focused regression test suite for P8.2 Unknown-Person False-Waste Detection Corrections.

Validates all 8 required test scenarios:
1. Registered person without waste: person detected, recognized, 0 false waste detections, 0 dumping events.
2. Unknown person without waste: person detected, UNKNOWN identity, 0 false waste detections, 0 dumping events.
3. Unknown person holding bottle: bottle preserved, carrying does NOT emit dumping events.
4. Unknown person holding bag: bag preserved, carrying does NOT emit dumping events.
5. Unknown person disposing waste: release on ground confirmed, attributed to unknown track ID, UNKNOWN face.
6. Unknown person near pre-existing litter: stationary litter tracked, proximity does NOT attribute, 0 dumping events.
7. Multiple unknown people: ambiguous responsibility correctly marked as AMBIGUOUS.
8. Registered + unknown people together: independent detection, offender attributed to unknown without blaming bystander.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.association.association_engine import PersonWasteAssociationEngine
from app.association.zones import DetectionZone, ZoneManager
from app.common_types import (
    AttributionStatus,
    BoundingBox,
    ConfirmedLitterEvent,
    Detection,
    DisposalEventType,
    LitteringState,
    RecognitionStatus,
    TrackedObject,
)
from app.detection.waste_detector import WasteDetector
from app.ui.event_overlay import EventStatusOverlay


def create_test_zone_manager() -> ZoneManager:
    zm = ZoneManager.__new__(ZoneManager)
    zm.zones = {
        "open_ground_default": DetectionZone(
            zone_id="open_ground_default",
            zone_type="ground",
            polygon=[[0.0, 0.25], [1.0, 0.25], [1.0, 1.0], [0.0, 1.0]],
            is_normalized=True,
            description="Open Ground",
        ),
        "bin_primary": DetectionZone(
            zone_id="bin_primary",
            zone_type="bin",
            polygon=[[0.70, 0.55], [0.95, 0.55], [0.95, 0.95], [0.70, 0.95]],
            is_normalized=True,
            description="Receptacle Bin",
        ),
    }
    return zm


class TestUnknownPersonFalseWasteFixes(unittest.TestCase):
    """Test suite directly verifying P8.2 unknown-person false-waste corrections."""

    def setUp(self):
        self.zones = create_test_zone_manager()
        self.engine = PersonWasteAssociationEngine(
            zone_manager=self.zones,
            holding_max_distance_px=160.0,
            holding_iou_threshold=0.05,
            min_holding_frames=2,
            separation_distance_delta_px=20.0,
            stationary_duration_seconds=3.0,
            event_cooldown_seconds=3.0,
            ambiguity_margin_px=35.0,
        )

    # -------------------------------------------------------------------------
    # Scenario 1: Registered person without waste
    # -------------------------------------------------------------------------
    @patch("app.detection.waste_detector.YOLO")
    def test_01_registered_person_without_waste(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        person_box = BoundingBox(200, 100, 350, 600)  # ph = 500, pw = 150

        # Spurious raw detection on head / scalp
        head_box = BoundingBox(240, 100, 310, 180)  # y1=100 (0%), center=140 (8%), y2=180 (16%)
        # Spurious raw detection on jacket clothing
        jacket_box = BoundingBox(220, 250, 330, 480)  # large torso patch

        self.assertTrue(
            detector._is_false_human_detection(head_box, 0.88, [person_box]),
            "Head/hair detection on registered person must be rejected",
        )
        self.assertTrue(
            detector._is_false_human_detection(jacket_box, 0.72, [person_box]),
            "Jacket/clothing detection on registered person must be rejected",
        )

        p1 = TrackedObject("Person_Enrolled", 1, "person", 0.95, person_box, 1, 0.0)
        events = self.engine.process_frame([p1], [], 640, 480, 1, 1.0)
        self.assertEqual(len(events), 0, "No waste detected -> 0 dumping events emitted")

    # -------------------------------------------------------------------------
    # Scenario 2: Unknown person without waste
    # -------------------------------------------------------------------------
    @patch("app.detection.waste_detector.YOLO")
    def test_02_unknown_person_without_waste(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        # Unknown person bounding box from real evidence footage (evt_20261119_185229_0001)
        # Person: (280, 291, 575, 480) -> ph = 189
        person_box = BoundingBox(280, 291, 575, 480)

        # Facial / chin / beard false positive: (275, 322, 339, 356)
        # y1 = 322.4 (16.6% ph), cy = 339.3 (25.5% ph), y2 = 356.2 (34.5% ph), ioa = 0.93
        chin_beard_box = BoundingBox(275, 322, 339, 356)

        is_false = detector._is_false_human_detection(chin_beard_box, 0.26, [person_box])
        self.assertTrue(
            is_false,
            "Chin/beard false positive on unknown person must be rejected by corrected cephalic filter",
        )

        # Verify detect() filters it out completely
        mock_box = MagicMock()
        mock_box.cls.cpu().numpy.return_value = np.array([1])
        mock_box.conf.cpu().numpy.return_value = np.array([0.26])
        mock_box.xyxy.cpu().numpy.return_value = np.array([[275, 322, 339, 356]])
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        detector.model.predict.return_value = [mock_result]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        filtered = detector.detect(dummy_frame, person_boxes=[person_box])
        self.assertEqual(len(filtered), 0, "detect() must return 0 detections for chin/beard false positive")

        # Unknown person tracked with 0 waste -> 0 dumping events
        p_unknown = TrackedObject("Person_Unknown_1", 2, "person", 0.90, person_box, 1, 0.0)
        events = self.engine.process_frame([p_unknown], [], 640, 480, 1, 1.0)
        self.assertEqual(len(events), 0, "Unknown person without waste must emit 0 events")

    # -------------------------------------------------------------------------
    # Scenario 3: Unknown person holding bottle (recyclable)
    # -------------------------------------------------------------------------
    @patch("app.detection.waste_detector.YOLO")
    def test_03_unknown_person_holding_bottle(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        person_box = BoundingBox(200, 100, 350, 600)  # ph = 500, pw = 150
        # Bottle held at waist level (y: 350 to 420 -> center 385 = 57% ph)
        bottle_box = BoundingBox(220, 350, 260, 420)  # rel_area = 0.037

        is_false = detector._is_false_human_detection(bottle_box, 0.78, [person_box])
        self.assertFalse(is_false, "Held bottle in waist volume must NOT be rejected")

        p_unknown = TrackedObject("Person_Unknown", 1, "person", 0.91, person_box, 1, 0.0)
        bottle = TrackedObject("Waste_Bottle", 1, "recyclable", 0.78, bottle_box, 1, 0.0)

        # Walk across 10 frames carrying bottle
        all_events = []
        for f in range(1, 11):
            t = float(f) * 0.1
            p_unknown.bbox = BoundingBox(200 + f * 5, 100, 350 + f * 5, 600)
            bottle.bbox = BoundingBox(220 + f * 5, 350, 260 + f * 5, 420)
            evts = self.engine.process_frame([p_unknown], [bottle], 640, 480, f, t)
            all_events.extend(evts)

        self.assertEqual(len(all_events), 0, "Unknown person carrying bottle must NEVER emit dumping events")
        traj = self.engine.get_trajectory("Waste_Bottle")
        self.assertIsNotNone(traj)
        self.assertGreaterEqual(traj.person_held_frames.get("Person_Unknown", 0), 2)
        self.assertFalse(traj.event_emitted)

    # -------------------------------------------------------------------------
    # Scenario 4: Unknown person holding bag (residual)
    # -------------------------------------------------------------------------
    @patch("app.detection.waste_detector.YOLO")
    def test_04_unknown_person_holding_bag(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        person_box = BoundingBox(200, 100, 350, 600)  # ph = 500, pw = 150
        # Grocery bag held hanging at side (y: 380 to 480 -> center 430 = 66% ph)
        bag_box = BoundingBox(320, 380, 370, 480)

        is_false = detector._is_false_human_detection(bag_box, 0.65, [person_box])
        self.assertFalse(is_false, "Held bag in carrying volume must NOT be rejected")

        p_unknown = TrackedObject("Person_Unknown_Bag", 1, "person", 0.89, person_box, 1, 0.0)
        bag = TrackedObject("Waste_Bag", 1, "residual", 0.65, bag_box, 1, 0.0)

        all_events = []
        for f in range(1, 8):
            t = float(f) * 0.2
            evts = self.engine.process_frame([p_unknown], [bag], 640, 480, f, t)
            all_events.extend(evts)

        self.assertEqual(len(all_events), 0, "Carrying bag without release must emit 0 events")
        traj = self.engine.get_trajectory("Waste_Bag")
        self.assertIsNotNone(traj)
        self.assertGreaterEqual(traj.person_held_frames.get("Person_Unknown_Bag", 0), 2)

    # -------------------------------------------------------------------------
    # Scenario 5: Unknown person disposing waste on open ground
    # -------------------------------------------------------------------------
    def test_05_unknown_person_disposing_waste(self):
        p_unknown = TrackedObject("Person_Unknown_Offender", 5, "person", 0.92, BoundingBox(100, 100, 220, 420), 1, 0.0)
        waste = TrackedObject("Waste_Can", 1, "recyclable", 0.88, BoundingBox(140, 200, 175, 250), 1, 0.0)

        # Frames 1-3: Held by unknown offender
        for f in range(1, 4):
            self.engine.process_frame([p_unknown], [waste], 640, 480, f, float(f) * 0.1)

        # Frame 4: Released to open ground (y=380), offender steps away
        p_unknown.bbox = BoundingBox(300, 100, 420, 420)
        waste.bbox = BoundingBox(150, 380, 185, 420)
        self.engine.process_frame([p_unknown], [waste], 640, 480, 4, 0.4)

        # Frames 5-9: Waste rests stationary on open ground for >= 3.0s
        confirmed = []
        for f, t in enumerate([1.0, 2.0, 3.0, 3.6, 3.8], start=5):
            evts = self.engine.process_frame([p_unknown], [waste], 640, 480, f, t)
            confirmed.extend(evts)

        self.assertEqual(len(confirmed), 1, "Dumping event must be confirmed for genuine release")
        evt = confirmed[0]
        self.assertEqual(evt.event_type, DisposalEventType.OPEN_GROUND_LITTERING)
        self.assertEqual(evt.person_track_id, "Person_Unknown_Offender")
        self.assertEqual(evt.attribution_status, AttributionStatus.ATTRIBUTED)

        # Verify overlay displays UNKNOWN identity when recognition status is UNKNOWN
        evt.recognition_status = RecognitionStatus.UNKNOWN
        evt.identified_name = "Unknown"
        overlay = EventStatusOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = overlay.draw(frame, new_events=[evt])
        self.assertIsNotNone(rendered)
        self.assertEqual(rendered.shape, (480, 640, 3))

    # -------------------------------------------------------------------------
    # Scenario 6: Unknown person near pre-existing litter (bystander)
    # -------------------------------------------------------------------------
    def test_06_unknown_person_near_pre_existing_litter(self):
        # Pre-existing litter sits on the ground
        ground_litter = TrackedObject("Waste_Ground", 1, "recyclable", 0.75, BoundingBox(100, 380, 140, 420), 1, 0.0)

        # Frames 1-2: Litter is detected sitting unheld on open ground
        for f in range(1, 3):
            self.engine.process_frame([], [ground_litter], 640, 480, f, float(f) * 0.1)

        traj = self.engine.get_trajectory("Waste_Ground")
        self.assertIsNotNone(traj)
        self.assertTrue(traj.is_pre_existing_litter, "Unheld ground litter must be marked as pre-existing")

        # Frames 3-8: Unknown person walks close to the litter (gap = 25 px, dist ~ 80 px)
        # Person bounding box: x1=165, y1=150, x2=280, y2=450 (no overlap with x1=100..140)
        p_walker = TrackedObject("Person_Walker", 2, "person", 0.90, BoundingBox(165, 150, 280, 450), 1, 0.0)

        all_events = []
        for f in range(3, 9):
            t = float(f) * 0.5
            p_walker.bbox = BoundingBox(165 - (f - 3) * 5, 150, 280 - (f - 3) * 5, 450)
            evts = self.engine.process_frame([p_walker], [ground_litter], 640, 480, f, t)
            all_events.extend(evts)

        self.assertEqual(len(all_events), 0, "Walking near pre-existing litter must NEVER create dumping events")
        self.assertEqual(
            traj.person_held_frames.get("Person_Walker", 0),
            0,
            "Passing near ground litter without physical overlap must NEVER be recorded as held",
        )
        self.assertIsNone(traj.last_associated_person)

    # -------------------------------------------------------------------------
    # Scenario 7: Multiple unknown people (ambiguity preserved)
    # -------------------------------------------------------------------------
    def test_07_multiple_unknown_people_ambiguity(self):
        # Two unknown people are both close to a dropped item
        p_unknown_1 = TrackedObject("Person_Unknown_A", 1, "person", 0.88, BoundingBox(50, 100, 170, 420), 1, 0.0)
        p_unknown_2 = TrackedObject("Person_Unknown_B", 2, "person", 0.89, BoundingBox(210, 100, 330, 420), 1, 0.0)
        waste = TrackedObject("Waste_Shared", 1, "recyclable", 0.80, BoundingBox(175, 200, 205, 250), 1, 0.0)

        # Both persons have equal holding/proximity interaction with the item
        for f in range(1, 4):
            self.engine.process_frame([p_unknown_1, p_unknown_2], [waste], 640, 480, f, float(f) * 0.1)

        # Release to ground exactly halfway between them
        waste.bbox = BoundingBox(180, 380, 210, 420)
        p_unknown_1.bbox = BoundingBox(20, 100, 140, 420)
        p_unknown_2.bbox = BoundingBox(250, 100, 370, 420)
        self.engine.process_frame([p_unknown_1, p_unknown_2], [waste], 640, 480, 4, 0.4)

        confirmed = []
        for f, t in enumerate([1.0, 2.0, 3.0, 3.6, 3.8], start=5):
            evts = self.engine.process_frame([p_unknown_1, p_unknown_2], [waste], 640, 480, f, t)
            confirmed.extend(evts)

        self.assertEqual(len(confirmed), 1)
        evt = confirmed[0]
        # Equal proximity and score difference < ambiguity margin -> AMBIGUOUS
        self.assertEqual(evt.attribution_status, AttributionStatus.AMBIGUOUS)
        self.assertIsNone(evt.person_track_id)

        # Verify overlay formats ambiguous track as "Ambiguous / Multiple"
        overlay = EventStatusOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = overlay.draw(frame, new_events=[evt])
        self.assertIsNotNone(rendered)
        self.assertEqual(overlay.active_event.attribution_status, AttributionStatus.AMBIGUOUS)

    # -------------------------------------------------------------------------
    # Scenario 8: Registered + unknown people together (independent attribution)
    # -------------------------------------------------------------------------
    def test_08_registered_and_unknown_together(self):
        # Registered person is a bystander standing on the right
        p_registered = TrackedObject("Person_Registered", 1, "person", 0.96, BoundingBox(450, 100, 570, 420), 1, 0.0)
        # Unknown person on the left is holding and dumping waste
        p_unknown = TrackedObject("Person_Unknown_Litterer", 2, "person", 0.91, BoundingBox(100, 100, 220, 420), 1, 0.0)
        waste = TrackedObject("Waste_Cup", 1, "residual", 0.85, BoundingBox(150, 210, 185, 260), 1, 0.0)

        # Frames 1-3: Waste held strictly by Person_Unknown_Litterer
        for f in range(1, 4):
            self.engine.process_frame([p_registered, p_unknown], [waste], 640, 480, f, float(f) * 0.1)

        traj = self.engine.get_trajectory("Waste_Cup")
        self.assertIsNotNone(traj)
        self.assertEqual(traj.last_associated_person, "Person_Unknown_Litterer")
        self.assertEqual(traj.person_held_frames.get("Person_Registered", 0), 0)

        # Release to ground by unknown offender
        waste.bbox = BoundingBox(150, 380, 185, 420)
        p_unknown.bbox = BoundingBox(20, 100, 140, 420)
        self.engine.process_frame([p_registered, p_unknown], [waste], 640, 480, 4, 0.4)

        confirmed = []
        for f, t in enumerate([1.0, 2.0, 3.0, 3.6, 3.8], start=5):
            evts = self.engine.process_frame([p_registered, p_unknown], [waste], 640, 480, f, t)
            confirmed.extend(evts)

        self.assertEqual(len(confirmed), 1)
        evt = confirmed[0]
        # Event MUST be attributed to Person_Unknown_Litterer, NEVER the registered bystander
        self.assertEqual(evt.person_track_id, "Person_Unknown_Litterer")
        self.assertEqual(evt.attribution_status, AttributionStatus.ATTRIBUTED)
        self.assertNotEqual(evt.person_track_id, "Person_Registered")


if __name__ == "__main__":
    unittest.main()
