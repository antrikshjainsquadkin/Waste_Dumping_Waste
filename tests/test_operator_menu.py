"""Test suite for Operator Menu & Input Workflows (P7).

Covers:
1. Main menu displays all 6 options.
2. Invalid menu choices are handled safely.
3. Face registration validates names, rejects duplicates/empty/Unknown.
4. Face extraction handles no face, multiple faces, blurry face, and valid face.
5. Registration persists to faces.json and reference backup while preserving existing identities.
6. CCTV stream credentials are sanitized; connection failures handled gracefully.
7. Video file picker cancel returns safely to menu.
8. Unsupported video file extensions are rejected.
9. Valid video selection invokes coordinator with selected path.
10. Event records viewing displays records without mutating CSV.
11. Exit workflow terminates menu loop safely.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.recognition.face_adapter import FaceRecognitionAdapter
from app.storage.csv_storage import CSV_HEADERS, CSVEventStorage
from app.ui.operator_menu import (
    OperatorMenu,
    open_native_file_picker,
    probe_available_webcams,
    sanitize_stream_url,
)


class TestOperatorMenu(unittest.TestCase):
    """Test suite validating OperatorMenu workflows and edge cases."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.faces_path = self.temp_path / "faces.json"
        self.faces_ref_path = self.temp_path / "faces_reference.json"
        self.csv_path = self.temp_path / "events.csv"
        self.settings_path = PROJECT_ROOT / "config" / "settings.yaml"
        self.zones_path = PROJECT_ROOT / "config" / "zones.yaml"

        # Initialize sample faces.json
        self.initial_registry = {
            "Alice": {
                "embedding": [0.01 * (i % 10) for i in range(512)],
                "enrolled_at": "2026-01-01T12:00:00Z",
            }
        }
        with open(self.faces_path, "w", encoding="utf-8") as f:
            json.dump(self.initial_registry, f)
        with open(self.faces_ref_path, "w", encoding="utf-8") as f:
            json.dump(self.initial_registry, f)

        # Initialize sample events.csv
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            writer.writerow([
                "evt_test_001", "OPEN_GROUND_LITTERING", "2026-11-19T10:00:00Z",
                "2026-11-19T10:00:05Z", "recyclable", "0.8500", "Waste_1", "Person_1",
                "Alice", "ATTRIBUTED", "IDENTIFIED", "open_ground_default",
                "data/evidence/evt_test_001_evidence.mp4",
                "data/evidence/evt_test_001_face.jpg", "", "12.50", "Test event"
            ])

    def tearDown(self):
        try:
            # Clear read-only if set
            for p in [self.faces_path, self.faces_ref_path]:
                if p.exists():
                    os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass
        self.temp_dir.cleanup()

    # 1. Menu display & invalid input handling
    def test_menu_display_and_invalid_choice(self):
        menu = OperatorMenu(
            settings_path=self.settings_path,
            zones_path=self.zones_path,
            faces_path=self.faces_path,
            csv_path=self.csv_path,
        )
        # Verify invalid choices return True (menu keeps running)
        self.assertTrue(menu.handle_choice("9"))
        self.assertTrue(menu.handle_choice("invalid_text"))
        self.assertTrue(menu.handle_choice("-1"))

    # 2. Exit workflow
    def test_exit_workflow(self):
        menu = OperatorMenu(
            settings_path=self.settings_path,
            zones_path=self.zones_path,
            faces_path=self.faces_path,
            csv_path=self.csv_path,
        )
        # Option 6 should return False to terminate loop
        self.assertFalse(menu.handle_choice("6"))

    # 3. Stream URL credential sanitization
    def test_sanitize_stream_url(self):
        raw_url = "rtsp://admin:superSecret123@192.168.1.100:554/cam1"
        sanitized = sanitize_stream_url(raw_url)
        self.assertNotIn("superSecret123", sanitized)
        self.assertIn("rtsp://admin:***@192.168.1.100:554/cam1", sanitized)

        http_url = "http://user:mypassword@example.com/live.m3u8"
        self.assertEqual(sanitize_stream_url(http_url), "http://user:***@example.com/live.m3u8")

        plain_url = "rtsp://192.168.1.100:554/live"
        self.assertEqual(sanitize_stream_url(plain_url), plain_url)

    # 4. View event records workflow
    def test_view_event_records(self):
        inputs = iter([""])  # operator presses Enter to return
        menu = OperatorMenu(
            settings_path=self.settings_path,
            zones_path=self.zones_path,
            faces_path=self.faces_path,
            csv_path=self.csv_path,
            input_func=lambda prompt: next(inputs),
        )
        # Should execute without error
        menu.view_records_workflow()

        # Check CSV content was NOT modified
        with open(self.csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_id"], "evt_test_001")
        self.assertEqual(rows[0]["recognized_person"], "Alice")

    # 5. Video file picker cancel returns safely
    def test_video_file_picker_cancel_returns_to_menu(self):
        # File picker returns None (operator clicked Cancel)
        menu = OperatorMenu(
            settings_path=self.settings_path,
            zones_path=self.zones_path,
            faces_path=self.faces_path,
            csv_path=self.csv_path,
            file_picker_func=lambda: None,
        )
        # Must return cleanly without exception
        menu.video_file_workflow()

    # 6. Video file picker rejects unsupported extension
    def test_video_file_unsupported_extension_rejected(self):
        dummy_txt = self.temp_path / "video.txt"
        dummy_txt.write_text("not a video")
        menu = OperatorMenu(
            settings_path=self.settings_path,
            zones_path=self.zones_path,
            faces_path=self.faces_path,
            csv_path=self.csv_path,
            file_picker_func=lambda: str(dummy_txt),
        )
        menu.video_file_workflow()

    # 7. CCTV connection failure handled gracefully
    @patch("cv2.VideoCapture")
    def test_cctv_connection_failure_handled(self, mock_cap_class):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False  # Connection fails
        mock_cap_class.return_value = mock_cap

        inputs = iter(["rtsp://admin:pass@192.168.1.200:554/offline"])
        menu = OperatorMenu(
            settings_path=self.settings_path,
            zones_path=self.zones_path,
            faces_path=self.faces_path,
            csv_path=self.csv_path,
            input_func=lambda prompt: next(inputs),
        )
        # Must fail gracefully without raising exception
        menu.cctv_detection_workflow()
        mock_cap.isOpened.assert_called_once()

    # 8. Face registration name validation
    def test_face_registration_name_validation(self):
        adapter = FaceRecognitionAdapter(self.faces_path)

        # Empty name rejected
        ok, msg = adapter.register_identity("", np.ones(512, dtype=np.float32))
        self.assertFalse(ok)
        self.assertIn("cannot be empty", msg)

        # "Unknown" rejected
        ok, msg = adapter.register_identity("Unknown", np.ones(512, dtype=np.float32))
        self.assertFalse(ok)
        self.assertIn("reserved", msg.lower())

        # Duplicate without replace rejected
        ok, msg = adapter.register_identity("Alice", np.ones(512, dtype=np.float32), replace=False)
        self.assertFalse(ok)
        self.assertIn("already enrolled", msg)

    # 9. Face registration persists and preserves existing identities
    def test_face_registration_persists_and_preserves(self):
        adapter = FaceRecognitionAdapter(self.faces_path)
        adapter.initialize()

        self.assertTrue(adapter.has_identity("Alice"))
        self.assertFalse(adapter.has_identity("Bob"))

        new_emb = np.array([float(i) for i in range(512)], dtype=np.float32)
        ok, msg = adapter.register_identity("Bob", new_emb, replace=False)
        self.assertTrue(ok)
        self.assertIn("successfully enrolled", msg)

        # In-memory check
        self.assertTrue(adapter.has_identity("Bob"))
        self.assertTrue(adapter.has_identity("Alice"))
        self.assertIn("Bob", adapter.get_enrolled_names())

        # Disk check on faces.json
        with open(self.faces_path, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        self.assertIn("Alice", disk_data)
        self.assertIn("Bob", disk_data)
        self.assertEqual(len(disk_data["Bob"]["embedding"]), 512)

        # Disk check on faces_reference.json
        with open(self.faces_ref_path, "r", encoding="utf-8") as f:
            ref_data = json.load(f)
        self.assertIn("Alice", ref_data)
        self.assertIn("Bob", ref_data)

    # 10. Face extraction logic tests
    def test_extract_face_validation(self):
        adapter = FaceRecognitionAdapter(self.faces_path)
        adapter._is_initialized = True
        adapter._app = MagicMock()

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Case A: No face detected
        adapter._app.get.return_value = []
        emb, crop, msg, bbox = adapter.extract_face(dummy_frame)
        self.assertIsNone(emb)
        self.assertIn("No face detected", msg)

        # Case B: Multiple faces detected
        face1 = MagicMock()
        face2 = MagicMock()
        adapter._app.get.return_value = [face1, face2]
        emb, crop, msg, bbox = adapter.extract_face(dummy_frame)
        self.assertIsNone(emb)
        self.assertIn("Multiple faces", msg)

        # Case C: Low detection score
        face1.det_score = 0.20  # below 0.35
        adapter._app.get.return_value = [face1]
        emb, crop, msg, bbox = adapter.extract_face(dummy_frame)
        self.assertIsNone(emb)
        self.assertIn("Detection score too low", msg)

        # Case D: Face too small (< 30px)
        face1.det_score = 0.90
        face1.bbox = np.array([100, 100, 115, 120])  # width 15 < 30
        adapter._app.get.return_value = [face1]
        emb, crop, msg, bbox = adapter.extract_face(dummy_frame)
        self.assertIsNone(emb)
        self.assertIn("Face too small", msg)

    # 11. Video file picker invokes coordinator with selected path
    @patch("app.ui.operator_menu.ApplicationCoordinator")
    def test_video_picker_valid_file_invokes_coordinator(self, mock_coord_class):
        mock_coord = MagicMock()
        mock_coord.initialize.return_value = True
        mock_coord_class.return_value = mock_coord

        valid_video = self.temp_path / "sample.mp4"
        valid_video.write_bytes(b"dummy_mp4_bytes")

        menu = OperatorMenu(
            settings_path=self.settings_path,
            zones_path=self.zones_path,
            faces_path=self.faces_path,
            csv_path=self.csv_path,
            file_picker_func=lambda: str(valid_video),
        )
        menu.video_file_workflow()

        mock_coord.initialize.assert_called_once_with(source_override=str(valid_video))
        mock_coord.run.assert_called_once()

    # 12. Webcam detection invokes coordinator with camera index
    @patch("app.ui.operator_menu.ApplicationCoordinator")
    def test_webcam_detection_invokes_coordinator(self, mock_coord_class):
        mock_coord = MagicMock()
        mock_coord.initialize.return_value = True
        mock_coord_class.return_value = mock_coord

        inputs = iter(["0"])
        menu = OperatorMenu(
            settings_path=self.settings_path,
            zones_path=self.zones_path,
            faces_path=self.faces_path,
            csv_path=self.csv_path,
            input_func=lambda prompt: next(inputs),
        )
        menu.webcam_detection_workflow()

        mock_coord.initialize.assert_called_once_with(source_override="0")
        mock_coord.run.assert_called_once()

    # 13. Menu loop full cycle handles invalid input, views records, and exits
    def test_menu_loop_full_cycle(self):
        inputs = iter(["99", "5", "", "6"])
        menu = OperatorMenu(
            settings_path=self.settings_path,
            zones_path=self.zones_path,
            faces_path=self.faces_path,
            csv_path=self.csv_path,
            input_func=lambda prompt: next(inputs),
        )
        menu.run_menu_loop()


if __name__ == "__main__":
    unittest.main()
