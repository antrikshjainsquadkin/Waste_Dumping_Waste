"""Focused verification test suite for waste detection fixes.

Validates:
1. A person's head/hair incorrectly classified as waste is rejected (even at high confidence).
2. Genuine waste carried in a person's hand is preserved.
3. A waste object partially occluded by a hand is preserved.
4. A waste object overlapping the person's bounding box is not suppressed by overlap alone.
5. A person holding waste without releasing it: detection is permitted, but no dumping event is created.
6. A genuine release: existing dumping-event logic triggers and attributes to thrower.
7. Existing bin-disposal behaviour remains unchanged.
"""

from __future__ import annotations

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
    Detection,
    DisposalEventType,
    TrackedObject,
)
from app.detection.waste_detector import WasteDetector


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


class TestWasteDetectionFixes(unittest.TestCase):
    """Test suite directly verifying fixes for head false positive and held waste false negatives."""

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

    # 1. Person's head/hair incorrectly classified as waste is rejected
    @patch("app.detection.waste_detector.YOLO")
    def test_head_hair_classified_as_waste_is_rejected(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        # Person bounding box: height 537, width 158
        person_box = BoundingBox(204, 326, 362, 863)

        # Hair / scalp false positive matching exact coordinates from real video (conf = 0.859)
        head_waste_box = BoundingBox(232, 326, 301, 435)

        is_false = detector._is_false_human_detection(head_waste_box, 0.859, [person_box])
        self.assertTrue(is_false, "Head/hair false positive must be rejected even at conf=0.86")

        # Verify detect() method strips it
        mock_box = MagicMock()
        mock_box.cls.cpu().numpy.return_value = np.array([1])  # recyclable
        mock_box.conf.cpu().numpy.return_value = np.array([0.859])
        mock_box.xyxy.cpu().numpy.return_value = np.array([[232, 326, 301, 435]])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        detector.model.predict.return_value = [mock_result]

        dummy_frame = np.zeros((864, 480, 3), dtype=np.uint8)
        filtered_dets = detector.detect(dummy_frame, person_boxes=[person_box])
        self.assertEqual(len(filtered_dets), 0, "detect() must return 0 detections for head/hair false positive")

    # 2. Genuine waste carried in a person's hand is preserved
    @patch("app.detection.waste_detector.YOLO")
    def test_genuine_waste_carried_in_hand_is_preserved(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        person_box = BoundingBox(204, 326, 362, 863)

        # Hand-held waste box (e.g. bottle or bag held at chest/hand level)
        held_waste_box = BoundingBox(188, 464, 236, 525)

        is_false = detector._is_false_human_detection(held_waste_box, 0.48, [person_box])
        self.assertFalse(is_false, "Hand-held waste must NOT be rejected by false human filter")

        # Verify detect() preserves it
        mock_box = MagicMock()
        mock_box.cls.cpu().numpy.return_value = np.array([2])  # residual
        mock_box.conf.cpu().numpy.return_value = np.array([0.48])
        mock_box.xyxy.cpu().numpy.return_value = np.array([[188, 464, 236, 525]])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        detector.model.predict.return_value = [mock_result]

        dummy_frame = np.zeros((864, 480, 3), dtype=np.uint8)
        filtered_dets = detector.detect(dummy_frame, person_boxes=[person_box])
        self.assertEqual(len(filtered_dets), 1, "detect() must preserve genuine held waste")
        self.assertEqual(filtered_dets[0].class_name, "residual")

    # 3. Waste partially occluded by hand is preserved
    @patch("app.detection.waste_detector.YOLO")
    def test_waste_partially_occluded_by_hand_is_preserved(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        person_box = BoundingBox(150, 100, 300, 500)
        # Partially occluded bottle held at waist level with lower confidence (0.32)
        occluded_bottle = BoundingBox(180, 320, 220, 380)

        is_false = detector._is_false_human_detection(occluded_bottle, 0.32, [person_box])
        self.assertFalse(is_false, "Partially occluded waste in hand must be preserved")

    # 4. Waste overlapping person's bounding box is preserved
    @patch("app.detection.waste_detector.YOLO")
    def test_waste_overlapping_person_box_is_preserved(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        person_box = BoundingBox(150, 100, 300, 500)
        # White bag held directly in front of body (IOA = 1.0, 100% inside person box)
        held_bag = BoundingBox(190, 250, 260, 340)

        is_false = detector._is_false_human_detection(held_bag, 0.50, [person_box])
        self.assertFalse(is_false, "Held bag 100% overlapping person box must NOT be rejected")

    # 5. Person holding waste without releasing: detection permitted, 0 dumping events created
    def test_carrying_waste_without_releasing_creates_no_dumping_event(self):
        # Person carries waste across 10 frames
        p1 = TrackedObject("Person_Carrier", 1, "person", 0.92, BoundingBox(100, 100, 220, 420), 1, 0.0)
        held_item = TrackedObject("Waste_Held", 1, "recyclable", 0.75, BoundingBox(150, 220, 185, 270), 1, 0.0)

        all_events = []
        for f_idx in range(1, 15):
            t = float(f_idx) * 0.3
            # Person and held item walk horizontally
            p1.bbox = BoundingBox(100 + f_idx * 10, 100, 220 + f_idx * 10, 420)
            held_item.bbox = BoundingBox(150 + f_idx * 10, 220, 185 + f_idx * 10, 270)
            events = self.engine.process_frame([p1], [held_item], 640, 480, f_idx, t)
            all_events.extend(events)

        self.assertEqual(len(all_events), 0, "Carrying waste without release must NEVER emit dumping events")
        traj = self.engine.get_trajectory("Waste_Held")
        self.assertIsNotNone(traj)
        self.assertGreaterEqual(traj.person_held_frames.get("Person_Carrier", 0), 2)
        self.assertFalse(traj.event_emitted)

    # 6. Genuine release triggers dumping event and attributes thrower
    def test_genuine_release_still_triggers_dumping_event(self):
        p1 = TrackedObject("Person_Thrower", 1, "person", 0.92, BoundingBox(100, 100, 220, 420), 1, 0.0)
        bottle = TrackedObject("Waste_Thrown", 1, "recyclable", 0.85, BoundingBox(150, 220, 185, 270), 1, 0.0)

        # Frames 1-3: Held by person
        for f_idx in range(1, 4):
            self.engine.process_frame([p1], [bottle], 640, 480, f_idx, float(f_idx) * 0.1)

        # Frame 4: Released to ground (y=380), person steps back
        p1.bbox = BoundingBox(40, 100, 140, 420)
        bottle.bbox = BoundingBox(220, 380, 250, 420)
        self.engine.process_frame([p1], [bottle], 640, 480, 4, 0.4)

        # Frames 5-10: Waste rests stationary on open ground for >= 3.0s
        confirmed = []
        for f_idx, t in enumerate([1.0, 2.0, 3.0, 3.6, 3.8], start=5):
            events = self.engine.process_frame([p1], [bottle], 640, 480, f_idx, t)
            confirmed.extend(events)

        self.assertEqual(len(confirmed), 1, "Must confirm genuine open ground disposal")
        self.assertEqual(confirmed[0].event_type, DisposalEventType.OPEN_GROUND_LITTERING)
        self.assertEqual(confirmed[0].person_track_id, "Person_Thrower")
        self.assertEqual(confirmed[0].attribution_status, AttributionStatus.ATTRIBUTED)

    # 7. Designated bin disposal remains functional
    def test_designated_bin_disposal_remains_functional(self):
        # Frame 1: Person holds waste near bin opening
        p1 = TrackedObject("Person_Good", 1, "person", 0.90, BoundingBox(450, 150, 550, 400), 1, 0.0)
        w1 = TrackedObject("Waste_Bin", 1, "recyclable", 0.90, BoundingBox(480, 180, 510, 220), 1, 0.0)
        self.engine.process_frame([p1], [w1], 640, 480, 1, 0.1)

        # Frame 2: Waste enters bin zone (x=[448..608], y=[264..456])
        w1.bbox = BoundingBox(490, 300, 520, 340)
        events = self.engine.process_frame([p1], [w1], 640, 480, 2, 0.2)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, DisposalEventType.BIN_DISPOSAL)
        self.assertEqual(events[0].person_track_id, "Person_Good")
        self.assertEqual(events[0].ground_zone_id, "bin_primary")


if __name__ == "__main__":
    unittest.main()
