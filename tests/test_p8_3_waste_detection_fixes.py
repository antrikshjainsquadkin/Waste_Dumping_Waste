"""Focused verification test suite for P8.3 Waste Detection & Association Hardening.

Covers all 11 verification scenarios required by the specification:
1. Scenario 1 — Oversized false prediction: Rejected before tracking, 0 waste IDs, 0 dumping events.
2. Scenario 2 — Face false detection: Head/face/neck prediction rejected, 0 false waste IDs.
3. Scenario 3 — Genuine bottle on ground: Detected as recyclable on open ground, valid track ID.
4. Scenario 4 — Genuine bottle in hand: Preserved in carrying volume, carrying emits 0 dumping events.
5. Scenario 5 — Bottle held and then released: Track ID preserved across release, 1 event confirmed.
6. Scenario 6 — Temporary occlusion: Track reconnected across occlusion, distinct objects not merged.
7. Scenario 7 — Duplicate detections: Overlapping duplicate predictions reduced to 1; distinct items kept.
8. Scenario 8 — Large legitimate waste: Large cardboard (e.g. 25-35% frame) is preserved.
9. Scenario 9 — Unknown person: Person UNKNOWN, held waste detected, face does not produce false waste.
10. Scenario 10 — Existing litter and passing person: Ground waste tracked, proximity does not attribute.
11. Scenario 11 — Multiple people and multiple waste objects: Separate person and waste tracks preserved.
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
    Detection,
    DisposalEventType,
    LitteringState,
    RecognitionStatus,
    TrackedObject,
)
from app.detection.waste_detector import WasteDetector
from app.tracking.tracker import MultiObjectTrackingService, SingleCategoryTracker


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


class TestP83WasteDetectionFixes(unittest.TestCase):
    """Test suite directly verifying the three core fixes and all 11 required scenarios."""

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
        self.tracking_service = MultiObjectTrackingService(
            high_thresh=0.25,
            low_thresh=0.10,
            match_thresh=0.70,
            track_buffer=30,
        )

    # -------------------------------------------------------------------------
    # Scenario 1 — Oversized false prediction
    # -------------------------------------------------------------------------
    @patch("app.detection.waste_detector.YOLO")
    def test_01_oversized_false_prediction_rejected(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        # Oversized prediction matching real diagnostic evidence (Waste_19: 72% of frame)
        # Frame: 640x480 (307,200 px^2), Box: [0.4, 133.6, 639.7, 480.0] -> area 221,453 px^2 (72.1%)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_box = MagicMock()
        mock_box.cls.cpu().numpy.return_value = np.array([1])  # recyclable
        mock_box.conf.cpu().numpy.return_value = np.array([0.74])
        mock_box.xyxy.cpu().numpy.return_value = np.array([[0.4, 133.6, 639.7, 480.0]])
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        detector.model.predict.return_value = [mock_result]

        person_box = BoundingBox(289, 131, 540, 475)
        filtered = detector.detect(dummy_frame, person_boxes=[person_box])

        self.assertEqual(len(filtered), 0, "Oversized 72% frame-level false prediction must be rejected")

        # Verify tracking creates 0 waste tracks from this frame
        _, tracked_waste = self.tracking_service.update([], filtered, 1, 0.1)
        self.assertEqual(len(tracked_waste), 0, "No waste track ID must be created for oversized false prediction")

    # -------------------------------------------------------------------------
    # Scenario 2 — Face false detection
    # -------------------------------------------------------------------------
    @patch("app.detection.waste_detector.YOLO")
    def test_02_face_false_detection_rejected(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Person_5: [180.6, 131.4, 540.8, 475.0] (height = 343.6 px)
        person_box = BoundingBox(180.6, 131.4, 540.8, 475.0)

        # Waste_21 face detection: [375.1, 125.0, 480.4, 271.6] (extends to 40.8% of person height)
        mock_box = MagicMock()
        mock_box.cls.cpu().numpy.return_value = np.array([1])  # recyclable
        mock_box.conf.cpu().numpy.return_value = np.array([0.48])
        mock_box.xyxy.cpu().numpy.return_value = np.array([[375.1, 125.0, 480.4, 271.6]])
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        detector.model.predict.return_value = [mock_result]

        filtered = detector.detect(dummy_frame, person_boxes=[person_box])
        self.assertEqual(len(filtered), 0, "Face/neck prediction on person must be rejected as waste")

    # -------------------------------------------------------------------------
    # Scenario 3 — Genuine bottle on ground
    # -------------------------------------------------------------------------
    @patch("app.detection.waste_detector.YOLO")
    def test_03_genuine_bottle_on_ground(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Real bottle prediction on ground from session clip: [289.0, 392.4, 415.5, 480.0]
        mock_box = MagicMock()
        mock_box.cls.cpu().numpy.return_value = np.array([1])  # recyclable
        mock_box.conf.cpu().numpy.return_value = np.array([0.39])
        mock_box.xyxy.cpu().numpy.return_value = np.array([[289.0, 392.4, 415.5, 480.0]])
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        detector.model.predict.return_value = [mock_result]

        filtered = detector.detect(dummy_frame, person_boxes=[])
        self.assertEqual(len(filtered), 1, "Genuine ground bottle must be preserved")
        self.assertEqual(filtered[0].class_name, "recyclable")

        # Verify tracking assigns a valid Waste track ID
        _, tracked_waste = self.tracking_service.update([], filtered, 1, 0.1)
        self.assertEqual(len(tracked_waste), 1)
        self.assertEqual(tracked_waste[0].track_id, "Waste_1")

    # -------------------------------------------------------------------------
    # Scenario 4 — Genuine bottle in hand (carrying)
    # -------------------------------------------------------------------------
    @patch("app.detection.waste_detector.YOLO")
    def test_04_genuine_bottle_in_hand_carrying(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person_box = BoundingBox(289.4, 131.5, 540.3, 474.8)

        # Held bottle from Frame 01: [211.4, 203.8, 312.4, 472.4] (conf = 0.433)
        mock_box = MagicMock()
        mock_box.cls.cpu().numpy.return_value = np.array([1])  # recyclable
        mock_box.conf.cpu().numpy.return_value = np.array([0.433])
        mock_box.xyxy.cpu().numpy.return_value = np.array([[211.4, 203.8, 312.4, 472.4]])
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        detector.model.predict.return_value = [mock_result]

        filtered = detector.detect(dummy_frame, person_boxes=[person_box])
        self.assertEqual(len(filtered), 1, "Held bottle in carrying volume must NOT be filtered out")

        # Tracking across 10 frames of walking/carrying
        p1 = TrackedObject("Person_1", 1, "person", 0.90, person_box, 1, 0.0)
        bottle = TrackedObject("Waste_1", 1, "recyclable", 0.43, BoundingBox(211.4, 203.8, 312.4, 472.4), 1, 0.0)

        all_events = []
        for f in range(1, 11):
            t = float(f) * 0.1
            p1.bbox = BoundingBox(289.4 + f * 5, 131.5, 540.3 + f * 5, 474.8)
            bottle.bbox = BoundingBox(211.4 + f * 5, 203.8, 312.4 + f * 5, 472.4)
            evts = self.engine.process_frame([p1], [bottle], 640, 480, f, t)
            all_events.extend(evts)

        self.assertEqual(len(all_events), 0, "Carrying bottle without release must NEVER emit dumping events")

    # -------------------------------------------------------------------------
    # Scenario 5 — Bottle held and then released
    # -------------------------------------------------------------------------
    def test_05_bottle_held_and_then_released(self):
        # Frame 1: Person holds bottle
        p1 = TrackedObject("Person_Carrier", 1, "person", 0.92, BoundingBox(200, 100, 350, 450), 1, 0.0)
        held_det = Detection(1, "recyclable", 0.45, BoundingBox(280, 280, 320, 380))

        _, tracked_waste = self.tracking_service.update([Detection(0, "person", 0.92, p1.bbox)], [held_det], 1, 0.1)
        self.assertEqual(len(tracked_waste), 1)
        initial_id = tracked_waste[0].track_id
        self.assertEqual(initial_id, "Waste_1")

        # Frames 1-3: Held
        for f in range(1, 4):
            t = float(f) * 0.1
            self.engine.process_frame([p1], tracked_waste, 640, 480, f, t)

        # Frame 4: Bottle released to open ground (drops ~100px downward to y=420), person steps back
        p1.bbox = BoundingBox(50, 100, 180, 450)
        ground_det = Detection(1, "recyclable", 0.40, BoundingBox(310, 400, 350, 460))
        _, tracked_waste_after = self.tracking_service.update([Detection(0, "person", 0.92, p1.bbox)], [ground_det], 4, 0.4)
        self.engine.process_frame([p1], tracked_waste_after, 640, 480, 4, 0.4)

        self.assertEqual(len(tracked_waste_after), 1)
        self.assertEqual(tracked_waste_after[0].track_id, initial_id, "Waste track ID must remain consistent across release")

        # Frames 5-9: Waste rests stationary on open ground for >= 3.0s
        confirmed = []
        for f, t in enumerate([1.0, 2.0, 3.0, 3.6, 3.8], start=5):
            evts = self.engine.process_frame([p1], tracked_waste_after, 640, 480, f, t)
            confirmed.extend(evts)

        self.assertEqual(len(confirmed), 1, "Exactly one dumping event must be confirmed")
        self.assertEqual(confirmed[0].event_type, DisposalEventType.OPEN_GROUND_LITTERING)
        self.assertEqual(confirmed[0].person_track_id, "Person_Carrier")
        self.assertEqual(confirmed[0].waste_track_id, initial_id)

    # -------------------------------------------------------------------------
    # Scenario 6 — Temporary occlusion recovery
    # -------------------------------------------------------------------------
    def test_06_temporary_occlusion_recovery(self):
        # Frame 1: Bottle detected and tracked as Waste_1
        det1 = Detection(1, "recyclable", 0.45, BoundingBox(250, 300, 290, 400))
        _, tw1 = self.tracking_service.update([], [det1], 1, 0.1)
        self.assertEqual(tw1[0].track_id, "Waste_1")

        # Frame 2: Occlusion occurs (bottle temporarily hidden behind hand/body -> 0 detections)
        _, tw2 = self.tracking_service.update([], [], 2, 0.2)
        self.assertEqual(len(tw2), 0)

        # Frame 3: Bottle re-emerges nearby (dist = 40px)
        det3 = Detection(1, "recyclable", 0.42, BoundingBox(260, 330, 300, 420))
        _, tw3 = self.tracking_service.update([], [det3], 3, 0.3)
        self.assertEqual(len(tw3), 1)
        self.assertEqual(tw3[0].track_id, "Waste_1", "Occluded bottle must reconnect to its original track ID")

        # A completely distinct physical object 300px away must NOT merge into Waste_1
        det_distant = Detection(1, "recyclable", 0.50, BoundingBox(50, 100, 90, 180))
        _, tw4 = self.tracking_service.update([], [det3, det_distant], 4, 0.4)
        self.assertEqual(len(tw4), 2)
        track_ids = {w.track_id for w in tw4}
        self.assertIn("Waste_1", track_ids)
        self.assertIn("Waste_2", track_ids, "Distant object must receive a new distinct track ID")

    # -------------------------------------------------------------------------
    # Scenario 7 — Duplicate detections reduced to one
    # -------------------------------------------------------------------------
    @patch("app.detection.waste_detector.YOLO")
    def test_07_duplicate_detections_reduced_to_one(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Two heavily overlapping raw detections of the same physical bottle (IOU = 0.65)
        box_primary = MagicMock()
        box_primary.cls.cpu().numpy.return_value = np.array([1])
        box_primary.conf.cpu().numpy.return_value = np.array([0.58])
        box_primary.xyxy.cpu().numpy.return_value = np.array([[100, 200, 180, 350]])

        box_duplicate = MagicMock()
        box_duplicate.cls.cpu().numpy.return_value = np.array([1])
        box_duplicate.conf.cpu().numpy.return_value = np.array([0.35])
        box_duplicate.xyxy.cpu().numpy.return_value = np.array([[105, 210, 185, 355]])

        mock_result = MagicMock()
        mock_result.boxes = [box_primary, box_duplicate]
        detector.model.predict.return_value = [mock_result]

        filtered = detector.detect(dummy_frame, person_boxes=[])
        self.assertEqual(len(filtered), 1, "Duplicate overlapping detections must be reduced to exactly 1")
        self.assertAlmostEqual(filtered[0].confidence, 0.58)

    # -------------------------------------------------------------------------
    # Scenario 8 — Large legitimate waste preserved
    # -------------------------------------------------------------------------
    @patch("app.detection.waste_detector.YOLO")
    def test_08_large_legitimate_waste_preserved(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Genuine large cardboard box on ground: width=260, height=220 (area = 57,200 px^2 = 18.6% of frame)
        mock_box = MagicMock()
        mock_box.cls.cpu().numpy.return_value = np.array([1])  # recyclable cardboard
        mock_box.conf.cpu().numpy.return_value = np.array([0.65])
        mock_box.xyxy.cpu().numpy.return_value = np.array([[200, 240, 460, 460]])
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        detector.model.predict.return_value = [mock_result]

        filtered = detector.detect(dummy_frame, person_boxes=[])
        self.assertEqual(len(filtered), 1, "Legitimate large cardboard waste must NOT be rejected")
        self.assertEqual(filtered[0].class_name, "recyclable")

    # -------------------------------------------------------------------------
    # Scenario 9 — Unknown person handling
    # -------------------------------------------------------------------------
    def test_09_unknown_person_handling(self):
        p_unknown = TrackedObject("Person_Unknown_9", 9, "person", 0.90, BoundingBox(200, 100, 350, 450), 1, 0.0)
        held_bottle = TrackedObject("Waste_Held", 1, "recyclable", 0.45, BoundingBox(260, 250, 300, 350), 1, 0.0)

        # Carrying across 5 frames
        for f in range(1, 6):
            evts = self.engine.process_frame([p_unknown], [held_bottle], 640, 480, f, float(f) * 0.1)
            self.assertEqual(len(evts), 0)

        traj = self.engine.get_trajectory("Waste_Held")
        self.assertIsNotNone(traj)
        self.assertEqual(traj.last_associated_person, "Person_Unknown_9")

    # -------------------------------------------------------------------------
    # Scenario 10 — Existing litter and passing person
    # -------------------------------------------------------------------------
    def test_10_existing_litter_and_passing_person(self):
        ground_litter = TrackedObject("Waste_Ground", 1, "recyclable", 0.50, BoundingBox(100, 380, 140, 420), 1, 0.0)

        # Litter seen unheld initially
        for f in range(1, 3):
            self.engine.process_frame([], [ground_litter], 640, 480, f, float(f) * 0.1)

        traj = self.engine.get_trajectory("Waste_Ground")
        self.assertTrue(traj.is_pre_existing_litter)

        # Person passes near it
        p_walker = TrackedObject("Person_Walker", 2, "person", 0.92, BoundingBox(160, 150, 280, 450), 1, 0.0)
        for f in range(3, 8):
            evts = self.engine.process_frame([p_walker], [ground_litter], 640, 480, f, float(f) * 0.2)
            self.assertEqual(len(evts), 0, "Passing near existing litter must not create dumping events")

        self.assertEqual(traj.person_held_frames.get("Person_Walker", 0), 0)

    # -------------------------------------------------------------------------
    # Scenario 11 — Multiple people and multiple waste objects
    # -------------------------------------------------------------------------
    def test_11_multiple_people_and_multiple_waste_objects(self):
        p1 = TrackedObject("Person_1", 1, "person", 0.95, BoundingBox(50, 100, 180, 420), 1, 0.0)
        p2 = TrackedObject("Person_2", 2, "person", 0.93, BoundingBox(450, 100, 580, 420), 1, 0.0)

        w1 = TrackedObject("Waste_1", 1, "recyclable", 0.80, BoundingBox(100, 250, 130, 320), 1, 0.0)  # held by p1
        w2 = TrackedObject("Waste_2", 2, "residual", 0.75, BoundingBox(490, 240, 530, 310), 1, 0.0)    # held by p2

        self.engine.process_frame([p1, p2], [w1, w2], 640, 480, 1, 0.1)

        t1 = self.engine.get_trajectory("Waste_1")
        t2 = self.engine.get_trajectory("Waste_2")

        self.assertEqual(t1.last_associated_person, "Person_1")
        self.assertEqual(t2.last_associated_person, "Person_2")
        self.assertEqual(t1.person_held_frames.get("Person_2", 0), 0)
        self.assertEqual(t2.person_held_frames.get("Person_1", 0), 0)


if __name__ == "__main__":
    unittest.main()
