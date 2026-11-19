"""Event attribution and evidence integrity tests for Waste_dumping_v2.

Verifies:
1. Single immutable event_id consistency across event object, video clip, face image, and CSV storage.
2. Attribution isolation: Face recognition searches only within the attributed offender's spatial history,
   preventing bystander face misattribution.
3. Face registry resilience: Auto-healing from reference backup when faces.json is truncated or corrupt.
4. Temporal window extraction: Precise pre-event (5s) and post-event (5s) window bounds.
5. Concurrent event isolation: Distinct events generate distinct immutable IDs and assets without collision.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.capture.frame_buffer import RollingFrameBuffer
from app.common_types import (
    AttributionStatus,
    BoundingBox,
    ConfirmedLitterEvent,
    DisposalEventType,
    RecognitionStatus,
)
from app.evidence.recorder import EvidenceRecorder
from app.recognition.face_adapter import FaceRecognitionAdapter
from app.storage.csv_storage import CSVEventStorage


class TestEventIntegrity(unittest.TestCase):
    """Test suite ensuring end-to-end evidence coupling and integrity."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.evidence_dir = self.temp_path / "evidence"
        self.csv_path = self.temp_path / "events.csv"
        self.faces_path = self.temp_path / "faces.json"
        self.faces_ref_path = self.temp_path / "faces_reference.json"

        # Populate a reference registry
        sample_registry = {
            "Alice": {"embedding": [1.0] + [0.0] * 511, "enrolled_at": "2026-01-01T00:00:00Z"},
            "Bob": {"embedding": [-1.0] + [0.0] * 511, "enrolled_at": "2026-01-01T00:00:00Z"},
        }
        with open(self.faces_ref_path, "w", encoding="utf-8") as f:
            json.dump(sample_registry, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_single_immutable_event_id_coupling(self):
        """Verify event_id binds the event object, video clip, face crop, and CSV row."""
        buffer = RollingFrameBuffer(max_seconds=12.0)
        t_base = 100.0
        for i in range(25):
            f = np.zeros((480, 640, 3), dtype=np.uint8)
            buffer.append(timestamp=t_base + i * 0.4, frame=f, frame_idx=i)

        recorder = EvidenceRecorder(
            evidence_dir=self.evidence_dir,
            frame_buffer=buffer,
            pre_event_seconds=5.0,
            post_event_seconds=5.0,
        )
        storage = CSVEventStorage(csv_path=self.csv_path)

        event_id = "evt_20261119_test_integrity_0001"
        event = ConfirmedLitterEvent(
            event_id=event_id,
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-19T10:00:00.000Z",
            estimated_release_timestamp=t_base + 5.0,
            confirmed_at="2026-11-19T10:00:03.000Z",
            confirmed_timestamp=t_base + 8.0,
            waste_category="residual",
            waste_confidence=0.88,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
            recognized_person="Alice",
            attribution_status=AttributionStatus.ATTRIBUTED,
            recognition_status=RecognitionStatus.IDENTIFIED,
        )

        mock_face_crop = np.full((100, 100, 3), 128, dtype=np.uint8)

        # Schedule and immediately flush clip
        recorder.schedule_clip(event=event, face_crop=mock_face_crop)
        flushed = recorder.flush_all()
        self.assertEqual(len(flushed), 1)

        # 1. Check returned paths carry the event_id
        self.assertIn(event_id, event.video_evidence_path)
        self.assertIn(event_id, event.face_snapshot_path)
        self.assertTrue(Path(event.video_evidence_path).exists())
        self.assertTrue(Path(event.face_snapshot_path).exists())

        # 2. Write to CSV and read back
        self.assertTrue(storage.append_event(event))

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 1)
        row = reader[0]

        # 3. Verify identical event_id in CSV row and referenced paths
        self.assertEqual(row["event_id"], event_id)
        self.assertEqual(row["video_evidence_path"], event.video_evidence_path)
        self.assertEqual(row["face_snapshot_path"], event.face_snapshot_path)
        self.assertEqual(row["recognized_person"], "Alice")
        self.assertEqual(row["attribution_status"], "ATTRIBUTED")

    def test_bystander_isolation_with_person_history(self):
        """Verify face recognition only inspects crops from the offender's bounding box history."""
        # Initialize adapter with dummy registry
        with open(self.faces_path, "w", encoding="utf-8") as f:
            json.dump({"Alice": {"embedding": [1.0] + [0.0] * 511}}, f)

        adapter = FaceRecognitionAdapter(
            faces_json_path=self.faces_path,
            similarity_threshold=0.5,
            min_sharpness=0.0,
        )
        adapter._load_registry()
        adapter._is_initialized = True

        mock_app = MagicMock()
        adapter._app = mock_app

        # Frame is 600x800 with distinct regions
        frame = np.full((600, 800, 3), 128, dtype=np.uint8)
        frames = [(100.0, frame, 1), (100.5, frame, 2)]

        # Track history only contains the offender's bounding box: [10, 10, 80, 80]
        person_history = [
            (1, 100.0, BoundingBox(10, 10, 80, 80)),
            (2, 100.5, BoundingBox(12, 10, 82, 80)),
        ]

        # Mock app.get() returns a detected face
        mock_face = MagicMock()
        mock_face.embedding = np.array([1.0] + [0.0] * 511, dtype=np.float32)
        mock_face.det_score = 0.95
        mock_face.bbox = np.array([5, 5, 55, 55], dtype=np.float32)
        mock_app.get.return_value = [mock_face]

        event = ConfirmedLitterEvent(
            event_id="evt_test_bystander",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-19T10:00:00.000Z",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-19T10:00:03.000Z",
            confirmed_timestamp=103.0,
            waste_category="residual",
            waste_confidence=0.88,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
            attribution_status=AttributionStatus.ATTRIBUTED,
        )

        name, conf, r_stat, crop = adapter.process_event(
            event=event,
            frames=frames,
            person_box=(10, 10, 80, 80),
            person_history=person_history,
        )

        self.assertEqual(r_stat, RecognitionStatus.IDENTIFIED)
        self.assertEqual(name, "Alice")
        self.assertIsNotNone(crop)

        # Crucial check: verify that app.get was called on cropped regions, NOT on the full 600x800 frame
        for call_args in mock_app.get.call_args_list:
            arg_img = call_args[0][0]
            # Height and width must be constrained to the cropped bounding box (approx 70x70 with margin)
            self.assertLess(arg_img.shape[0], 200)
            self.assertLess(arg_img.shape[1], 200)

    def test_faces_json_truncation_auto_recovery(self):
        """Verify FaceRecognitionAdapter auto-heals when faces.json is truncated to 0 bytes."""
        # Truncate faces.json to 0 bytes
        with open(self.faces_path, "w", encoding="utf-8") as f:
            f.write("")
        self.assertEqual(os.path.getsize(self.faces_path), 0)

        # Initialize adapter pointing to the empty faces.json, with reference file in same dir
        adapter = FaceRecognitionAdapter(
            faces_json_path=self.faces_path,
        )
        adapter._load_registry()

        # The adapter should have restored registry from faces_reference.json
        self.assertIn("Alice", adapter._registry)
        self.assertIn("Bob", adapter._registry)
        # The corrupted file should now be restored with valid JSON
        self.assertGreater(os.path.getsize(self.faces_path), 0)

    def test_temporal_window_bounds(self):
        """Verify that rolling frame buffer returns strictly [event_time - 5s, event_time + 5s]."""
        buffer = RollingFrameBuffer(max_seconds=12.0)
        start_t = 1000.0
        # Add 60 frames spaced at 0.2s (total 12 seconds: 1000.0 to 1011.8)
        for i in range(60):
            t = start_t + i * 0.2
            buffer.append(timestamp=t, frame=np.zeros((10, 10, 3), dtype=np.uint8), frame_idx=i)

        event_time = 1006.0
        window_frames = buffer.get_window(
            start_ts=event_time - 5.0,
            end_ts=event_time + 5.0,
        )

        # Window: 1001.0 to 1011.0. With 0.2s step, that's 51 frames (1001.0 to 1011.0 inclusive)
        self.assertGreater(len(window_frames), 45)
        self.assertLessEqual(len(window_frames), 55)
        # Check first and last frame timestamp
        self.assertGreaterEqual(window_frames[0][0], 1001.0)
        self.assertLessEqual(window_frames[-1][0], 1011.0)

    def test_concurrent_disposal_event_isolation(self):
        """Verify distinct sequential events maintain separate IDs and do not collide in storage."""
        storage = CSVEventStorage(csv_path=self.csv_path)

        event1 = ConfirmedLitterEvent(
            event_id="evt_001",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-19T10:00:01.000Z",
            estimated_release_timestamp=1.0,
            confirmed_at="2026-11-19T10:00:04.000Z",
            confirmed_timestamp=4.0,
            waste_category="organic",
            waste_confidence=0.8,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
            recognized_person="PersonA",
            attribution_status=AttributionStatus.ATTRIBUTED,
            recognition_status=RecognitionStatus.IDENTIFIED,
            video_evidence_path="vid_001.mp4",
            face_snapshot_path="face_001.jpg",
        )

        event2 = ConfirmedLitterEvent(
            event_id="evt_002",
            event_type=DisposalEventType.BIN_DISPOSAL,
            estimated_release_time="2026-11-19T10:00:03.000Z",
            estimated_release_timestamp=3.0,
            confirmed_at="2026-11-19T10:00:06.000Z",
            confirmed_timestamp=6.0,
            waste_category="recyclable",
            waste_confidence=0.9,
            waste_track_id="Waste_2",
            person_track_id="Person_2",
            recognized_person="Unknown",
            attribution_status=AttributionStatus.ATTRIBUTED,
            recognition_status=RecognitionStatus.UNKNOWN,
            video_evidence_path="vid_002.mp4",
            face_snapshot_path="face_002.jpg",
        )

        self.assertTrue(storage.append_event(event1))
        self.assertTrue(storage.append_event(event2))

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["event_id"], "evt_001")
        self.assertEqual(rows[0]["event_type"], "OPEN_GROUND_LITTERING")
        self.assertEqual(rows[0]["recognized_person"], "PersonA")
        self.assertEqual(rows[1]["event_id"], "evt_002")
        self.assertEqual(rows[1]["event_type"], "BIN_DISPOSAL")
        self.assertEqual(rows[1]["recognized_person"], "Unknown")


class TestFaceRecognitionConfigurationAndPolicy(unittest.TestCase):
    """Regression tests verifying baseline recognition thresholds and matching policies."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.faces_path = Path(self.temp_dir.name) / "faces.json"
        registry = {
            "Alice": {"embedding": [1.0] + [0.0] * 511},
            "Bob": {"embedding": [0.0, 1.0] + [0.0] * 510},
        }
        with open(self.faces_path, "w", encoding="utf-8") as f:
            json.dump(registry, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_baseline_configuration_constants(self):
        """Verify constructor defaults and settings.yaml match the authorized baseline."""
        import yaml
        settings_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(settings_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        rec_cfg = cfg["recognition"]

        # 1. Verify config/settings.yaml baseline values
        self.assertEqual(rec_cfg["min_det_score"], 0.35)
        self.assertEqual(rec_cfg["min_face_size_px"], 30)
        self.assertEqual(rec_cfg["similarity_threshold"], 0.50)
        self.assertEqual(rec_cfg["ambiguity_margin"], 0.03)
        self.assertEqual(rec_cfg["min_sharpness"], 12.0)

        # 2. Verify FaceRecognitionAdapter constructor defaults
        adapter = FaceRecognitionAdapter(faces_json_path=self.faces_path)
        self.assertEqual(adapter.min_det_score, 0.35)
        self.assertEqual(adapter.min_face_size, 30)
        self.assertEqual(adapter.sim_threshold, 0.50)
        self.assertEqual(adapter.ambiguity_margin, 0.03)
        self.assertEqual(adapter.min_sharpness, 12.0)

    def _create_mock_adapter(self, sim_thresh=0.50, ambiguity_margin=0.03, min_det=0.35, min_size=30, min_sharp=12.0):
        adapter = FaceRecognitionAdapter(
            faces_json_path=self.faces_path,
            similarity_threshold=sim_thresh,
            ambiguity_margin=ambiguity_margin,
            min_det_score=min_det,
            min_face_size_px=min_size,
            min_sharpness=min_sharp,
        )
        adapter._load_registry()
        adapter._is_initialized = True
        adapter._app = MagicMock()
        return adapter

    def _create_event(self):
        return ConfirmedLitterEvent(
            event_id="evt_test_recog_policy",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-19T10:00:00.000Z",
            estimated_release_timestamp=100.0,
            confirmed_at="2026-11-19T10:00:03.000Z",
            confirmed_timestamp=103.0,
            waste_category="residual",
            waste_confidence=0.88,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
            attribution_status=AttributionStatus.ATTRIBUTED,
        )

    def test_valid_enrolled_face(self):
        """Enrolled identity with similarity >= 0.50 is correctly identified."""
        adapter = self._create_mock_adapter()
        event = self._create_event()

        # Mock face matching Alice (sim = 0.95 >= 0.50)
        mock_face = MagicMock()
        mock_face.embedding = np.array([0.95, 0.05] + [0.0] * 510, dtype=np.float32)
        mock_face.det_score = 0.85
        mock_face.bbox = np.array([10, 10, 60, 60], dtype=np.float32)
        adapter._app.get.return_value = [mock_face]

        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        frames = [(100.0, frame, 1)]
        person_box = (5, 5, 100, 100)

        name, sim, status, crop = adapter.process_event(event, frames, person_box=person_box)
        self.assertEqual(status, RecognitionStatus.IDENTIFIED)
        self.assertEqual(name, "Alice")
        self.assertGreaterEqual(sim, 0.50)

    def test_unknown_face_below_threshold(self):
        """Unenrolled face with similarity < 0.50 is marked UNKNOWN."""
        adapter = self._create_mock_adapter()
        event = self._create_event()

        # Face orthogonal to both Alice and Bob (sim ~ 0.0 < 0.50)
        mock_face = MagicMock()
        mock_face.embedding = np.array([0.0, 0.0, 1.0] + [0.0] * 509, dtype=np.float32)
        mock_face.det_score = 0.80
        mock_face.bbox = np.array([10, 10, 60, 60], dtype=np.float32)
        adapter._app.get.return_value = [mock_face]

        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        frames = [(100.0, frame, 1)]
        person_box = (5, 5, 100, 100)

        name, sim, status, crop = adapter.process_event(event, frames, person_box=person_box)
        self.assertEqual(status, RecognitionStatus.UNKNOWN)
        self.assertEqual(name, "Unknown")
        self.assertLess(sim, 0.50)

    def test_low_quality_face_rejection(self):
        """Faces failing detection score (<0.35), size (<30px), or sharpness (<12.0) are rejected."""
        adapter = self._create_mock_adapter()
        event = self._create_event()
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        frames = [(100.0, frame, 1)]
        person_box = (5, 5, 100, 100)

        # 1. Detection score < 0.35
        mock_face_low_det = MagicMock()
        mock_face_low_det.det_score = 0.20
        mock_face_low_det.bbox = np.array([10, 10, 60, 60], dtype=np.float32)
        adapter._app.get.return_value = [mock_face_low_det]
        _, _, status, _ = adapter.process_event(event, frames, person_box=person_box)
        self.assertEqual(status, RecognitionStatus.NO_FACE)

        # 2. Face size < 30px (e.g. 20x20)
        mock_face_small = MagicMock()
        mock_face_small.det_score = 0.90
        mock_face_small.bbox = np.array([10, 10, 25, 25], dtype=np.float32)  # width 15, height 15
        adapter._app.get.return_value = [mock_face_small]
        _, _, status, _ = adapter.process_event(event, frames, person_box=person_box)
        self.assertEqual(status, RecognitionStatus.NO_FACE)

        # 3. Low sharpness (completely uniform blank crop has Laplacian var = 0 < 12.0)
        mock_face_blur = MagicMock()
        mock_face_blur.det_score = 0.90
        mock_face_blur.bbox = np.array([10, 10, 60, 60], dtype=np.float32)
        adapter._app.get.return_value = [mock_face_blur]
        blank_frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        _, _, status, _ = adapter.process_event(event, [(100.0, blank_frame, 1)], person_box=person_box)
        self.assertEqual(status, RecognitionStatus.NO_FACE)

    def test_ambiguous_matches_yield_unknown(self):
        """Top match within ambiguity margin (0.03) of second match is marked UNKNOWN."""
        adapter = self._create_mock_adapter()
        event = self._create_event()

        # Embedding has nearly equal similarity to Alice and Bob (diff < 0.03)
        # Alice is [1, 0, ...], Bob is [0, 1, ...]
        # Vector [0.707, 0.707, ...] has equal similarity 0.707 to both
        mock_face = MagicMock()
        mock_face.embedding = np.array([0.71, 0.70] + [0.0] * 510, dtype=np.float32)
        mock_face.det_score = 0.90
        mock_face.bbox = np.array([10, 10, 60, 60], dtype=np.float32)
        adapter._app.get.return_value = [mock_face]

        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        frames = [(100.0, frame, 1)]
        person_box = (5, 5, 100, 100)

        name, sim, status, crop = adapter.process_event(event, frames, person_box=person_box)
        self.assertEqual(status, RecognitionStatus.UNKNOWN)
        self.assertEqual(name, "Unknown")


if __name__ == "__main__":
    unittest.main()
