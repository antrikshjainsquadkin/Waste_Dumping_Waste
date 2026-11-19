"""Unit test suite for Waste_dumping_v2 core components."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.association.zones import DetectionZone, ZoneManager
from app.capture.frame_buffer import RollingFrameBuffer
from app.common_types import (
    AttributionStatus,
    BoundingBox,
    ConfirmedLitterEvent,
    Detection,
    DisposalEventType,
    RecognitionStatus,
)
from app.detection.waste_detector import WasteDetector
from app.storage.csv_storage import CSV_HEADERS, CSVEventStorage
from app.tracking.tracker import MultiObjectTrackingService, SingleCategoryTracker


class TestBoundingBox(unittest.TestCase):
    """Test geometry and overlap calculations."""

    def test_dimensions_and_center(self):
        box = BoundingBox(x1=10, y1=20, x2=50, y2=80)
        self.assertEqual(box.width, 40)
        self.assertEqual(box.height, 60)
        self.assertEqual(box.area, 2400)
        self.assertEqual(box.center, (30.0, 50.0))

    def test_iou_calculation(self):
        box1 = BoundingBox(0, 0, 10, 10)
        box2 = BoundingBox(5, 0, 15, 10)
        # Inter: 5x10 = 50. Union: 100 + 100 - 50 = 150. IoU: 50/150 = 1/3
        self.assertAlmostEqual(box1.iou(box2), 1.0 / 3.0, places=4)

    def test_ioa_calculation(self):
        box_inner = BoundingBox(2, 2, 8, 8)
        box_outer = BoundingBox(0, 0, 10, 10)
        self.assertAlmostEqual(box_inner.ioa(box_outer), 1.0)
        self.assertAlmostEqual(box_outer.ioa(box_inner), 36.0 / 100.0)


class TestRollingFrameBuffer(unittest.TestCase):
    """Test ring buffer capacity and window retrieval."""

    def test_buffer_retrieval(self):
        buf = RollingFrameBuffer(max_seconds=2.0, assumed_fps=10.0)
        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)

        for i in range(25):
            buf.append(timestamp=float(i) * 0.1, frame=dummy_img, frame_idx=i)

        self.assertLessEqual(len(buf), 20)
        # Query window between 1.5 and 2.5
        window = buf.get_window(1.5, 2.5)
        self.assertGreater(len(window), 0)
        for ts, _, _ in window:
            self.assertTrue(1.5 <= ts <= 2.5)


class TestTrackingService(unittest.TestCase):
    """Test ByteTrack multi-object tracking and namespace isolation."""

    def test_separate_namespaces(self):
        service = MultiObjectTrackingService()
        p_det = [Detection(0, "person", 0.90, BoundingBox(100, 100, 200, 300))]
        w_det = [Detection(1, "recyclable", 0.85, BoundingBox(120, 150, 140, 180))]

        p_tracks, w_tracks = service.update(p_det, w_det, frame_idx=1, timestamp=1.0)
        self.assertEqual(len(p_tracks), 1)
        self.assertEqual(len(w_tracks), 1)
        self.assertTrue(p_tracks[0].track_id.startswith("Person_"))
        self.assertTrue(w_tracks[0].track_id.startswith("Waste_"))

    def test_id_persistence_across_frames(self):
        service = MultiObjectTrackingService()
        p_det1 = [Detection(0, "person", 0.90, BoundingBox(100, 100, 200, 300))]
        p_tracks1, _ = service.update(p_det1, [], frame_idx=1, timestamp=1.0)
        pid1 = p_tracks1[0].track_id

        # Move person slightly
        p_det2 = [Detection(0, "person", 0.88, BoundingBox(105, 102, 205, 302))]
        p_tracks2, _ = service.update(p_det2, [], frame_idx=2, timestamp=1.033)
        pid2 = p_tracks2[0].track_id

        self.assertEqual(pid1, pid2, "Person track ID must persist across consecutive frames")


class TestNonWasteRejection(unittest.TestCase):
    """Test 3-phase false positive rejection filter."""

    @patch("app.detection.waste_detector.YOLO")
    def test_reject_full_body_match(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        person_box = BoundingBox(100, 100, 250, 450)
        # Waste box matching almost entire body
        waste_body_box = BoundingBox(105, 105, 245, 445)
        is_false = detector._is_false_human_detection(waste_body_box, 0.75, [person_box])
        self.assertTrue(is_false, "Should reject waste candidate that matches human body")

    @patch("app.detection.waste_detector.YOLO")
    def test_reject_head_hair(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        person_box = BoundingBox(100, 100, 250, 450)
        # Waste detection located on top crown of head / hair
        head_box = BoundingBox(145, 100, 205, 160)
        is_false = detector._is_false_human_detection(head_box, 0.85, [person_box])
        self.assertTrue(is_false, "Should reject false waste detection on human head/hair")

    @patch("app.detection.waste_detector.YOLO")
    def test_reject_large_torso_clothing(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        person_box = BoundingBox(100, 100, 250, 450)
        # Large clothing/shirt box covering majority of torso
        shirt_box = BoundingBox(105, 180, 245, 330)
        is_false = detector._is_false_human_detection(shirt_box, 0.55, [person_box])
        self.assertTrue(is_false, "Should reject large clothing/shirt detection on human torso")

    @patch("app.detection.waste_detector.YOLO")
    def test_preserve_genuine_held_waste(self, mock_yolo):
        detector = WasteDetector(
            model_path="models/waste/waste_detection_yolo11m.pt",
            verify_sha256=False,
        )
        person_box = BoundingBox(100, 100, 250, 450)
        # Case A: Waste object extending outward from hand (partially overlapping)
        held_bottle_outward = BoundingBox(230, 280, 275, 330)
        self.assertFalse(
            detector._is_false_human_detection(held_bottle_outward, 0.85, [person_box]),
            "Should preserve genuine waste extending outward from body",
        )

        # Case B: Waste object held directly in front of body/torso with moderate confidence
        held_bag_in_front = BoundingBox(140, 220, 180, 280)
        self.assertFalse(
            detector._is_false_human_detection(held_bag_in_front, 0.45, [person_box]),
            "Should preserve genuine waste held directly in front of person body",
        )


class TestCSVStorage(unittest.TestCase):
    """Test CSV schema and append operations."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "test_events.csv"
        self.storage = CSVEventStorage(self.csv_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schema_and_append(self):
        self.assertEqual(self.storage.total_records, 0)
        event = ConfirmedLitterEvent(
            event_id="evt_001",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-18T10:00:00Z",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-18T10:00:05Z",
            confirmed_timestamp=105.0,
            waste_category="recyclable",
            waste_confidence=0.92,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
            recognized_person="antriksh",
            attribution_status=AttributionStatus.ATTRIBUTED,
            recognition_status=RecognitionStatus.IDENTIFIED,
            ground_zone_id="open_ground_default",
            video_evidence_path="data/evidence/evt_001_evidence.mp4",
            face_snapshot_path="data/evidence/evt_001_face.jpg",
            notes="Test record",
        )
        ok = self.storage.append_event(event)
        self.assertTrue(ok)
        self.assertEqual(self.storage.total_records, 1)

        # Duplicate rejection
        dup_ok = self.storage.append_event(event)
        self.assertFalse(dup_ok, "Must reject duplicate event_id")
        self.assertEqual(self.storage.total_records, 1)


if __name__ == "__main__":
    unittest.main()
