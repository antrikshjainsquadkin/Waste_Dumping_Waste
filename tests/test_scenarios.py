"""Integration test suite covering all 15 core verification scenarios for Waste_dumping_v2."""

from __future__ import annotations

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

from app.association.association_engine import PersonWasteAssociationEngine
from app.association.zones import DetectionZone, ZoneManager
from app.capture.frame_buffer import RollingFrameBuffer
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
from app.evidence.recorder import EvidenceRecorder
from app.recognition.face_adapter import FaceRecognitionAdapter
from app.storage.csv_storage import CSVEventStorage
from app.tracking.tracker import MultiObjectTrackingService


def create_test_zone_manager() -> ZoneManager:
    """Create a ZoneManager with open ground and bin zones for testing."""
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


class TestAll15Scenarios(unittest.TestCase):
    """Test suite directly validating the 15 required functional scenarios."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "events.csv"
        self.evidence_dir = Path(self.temp_dir.name) / "evidence"
        self.storage = CSVEventStorage(self.csv_path)
        self.zones = create_test_zone_manager()
        self.engine = PersonWasteAssociationEngine(
            zone_manager=self.zones,
            holding_max_distance_px=160.0,
            holding_iou_threshold=0.05,
            min_holding_frames=2,
            separation_distance_delta_px=20.0,
            stationary_duration_seconds=3.0,
            event_cooldown_seconds=2.0,
            ambiguity_margin_px=35.0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scenario_01_known_person_throws_waste_on_open_ground(self):
        """Scenario 1: Known person throws waste on open ground -> Event saved with recognized identity."""
        # Frame 1-3: Person_1 holding recyclable bottle
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "recyclable", 0.88, BoundingBox(140, 220, 170, 270), 1, 0.0)

        for i in range(1, 4):
            t = float(i) * 0.1
            events = self.engine.process_frame([p1], [w1], 640, 480, i, t)
            self.assertEqual(len(events), 0, "No event while holding")

        # Frame 4: Released to ground (y=380), person moves away
        p1.bbox = BoundingBox(50, 100, 150, 350)
        w1.bbox = BoundingBox(250, 380, 280, 420)  # in open ground
        events = self.engine.process_frame([p1], [w1], 640, 480, 4, 0.4)
        self.assertEqual(len(events), 0, "Still accumulating stability")

        # Waste stays stationary on ground for > 3.0 seconds (from t=0.4 to t=3.6)
        confirmed = []
        for f_idx, t in enumerate([1.0, 2.0, 3.0, 3.5, 3.6], start=5):
            events = self.engine.process_frame([p1], [w1], 640, 480, f_idx, t)
            if events:
                confirmed.extend(events)

        self.assertEqual(len(confirmed), 1, "Must confirm exactly 1 open ground littering event")
        evt = confirmed[0]
        self.assertEqual(evt.event_type, DisposalEventType.OPEN_GROUND_LITTERING)
        self.assertEqual(evt.person_track_id, "Person_1")
        self.assertEqual(evt.attribution_status, AttributionStatus.ATTRIBUTED)

        # Mock FaceRecognitionAdapter returning enrolled identity 'antriksh'
        adapter = FaceRecognitionAdapter(PROJECT_ROOT / "data" / "faces.json")
        adapter._registry["antriksh"] = np.ones(512, dtype=np.float32) / np.sqrt(512)
        adapter._is_initialized = True
        adapter._app = MagicMock()

        dummy_face = MagicMock()
        dummy_face.det_score = 0.95
        dummy_face.bbox = np.array([10, 10, 50, 50])
        dummy_face.embedding = np.ones(512, dtype=np.float32)
        adapter._app.get.return_value = [dummy_face]

        dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Add high contrast gradient so Laplacian variance passes sharpness threshold
        dummy_frame[::2, ::2] = 255
        dummy_frame[1::2, 1::2] = 0

        name, sim, r_stat, crop = adapter.process_event(evt, [(3.6, dummy_frame, 10)], (10, 10, 80, 80))
        evt.recognized_person = name
        evt.recognition_status = r_stat

        self.assertEqual(evt.recognized_person, "antriksh")
        self.assertEqual(evt.recognition_status, RecognitionStatus.IDENTIFIED)

        ok = self.storage.append_event(evt)
        self.assertTrue(ok)
        self.assertEqual(self.storage.total_records, 1)

    def test_scenario_02_unknown_person_throws_waste_on_open_ground(self):
        """Scenario 2: Unknown person throws waste on open ground -> Event saved with Unknown."""
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "organic", 0.85, BoundingBox(140, 220, 170, 270), 1, 0.0)

        for i in range(1, 4):
            self.engine.process_frame([p1], [w1], 640, 480, i, float(i) * 0.1)

        # Separation and landing on ground
        p1.bbox = BoundingBox(30, 100, 130, 350)
        w1.bbox = BoundingBox(300, 400, 330, 440)

        confirmed = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            events = self.engine.process_frame([p1], [w1], 640, 480, f_idx, t)
            if events:
                confirmed.extend(events)

        self.assertEqual(len(confirmed), 1)
        evt = confirmed[0]

        # Face recognition matching unknown person (low similarity)
        adapter = FaceRecognitionAdapter(PROJECT_ROOT / "data" / "faces.json")
        adapter._registry["antriksh"] = np.ones(512, dtype=np.float32) / np.sqrt(512)
        adapter._is_initialized = True
        adapter._app = MagicMock()

        dummy_face = MagicMock()
        dummy_face.det_score = 0.90
        dummy_face.bbox = np.array([10, 10, 50, 50])
        # Orthogonal embedding -> similarity = 0.0 < 0.50
        dummy_face.embedding = np.zeros(512, dtype=np.float32)
        dummy_face.embedding[0] = 1.0
        adapter._app.get.return_value = [dummy_face]

        dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        dummy_frame[::2, ::2] = 255
        name, sim, r_stat, _ = adapter.process_event(evt, [(3.8, dummy_frame, 10)], (10, 10, 80, 80))
        evt.recognized_person = name
        evt.recognition_status = r_stat

        self.assertEqual(evt.recognized_person, "Unknown")
        self.assertEqual(evt.recognition_status, RecognitionStatus.UNKNOWN)
        self.storage.append_event(evt)
        self.assertEqual(self.storage.total_records, 1)

    def test_scenario_03_person_face_not_visible(self):
        """Scenario 3: Person's face is not visible -> Event still saved."""
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "residual", 0.85, BoundingBox(140, 220, 170, 270), 1, 0.0)

        for i in range(1, 4):
            self.engine.process_frame([p1], [w1], 640, 480, i, float(i) * 0.1)

        p1.bbox = BoundingBox(30, 100, 130, 350)
        w1.bbox = BoundingBox(300, 400, 330, 440)

        confirmed = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            events = self.engine.process_frame([p1], [w1], 640, 480, f_idx, t)
            if events:
                confirmed.extend(events)

        self.assertEqual(len(confirmed), 1)
        evt = confirmed[0]

        # No face detected in frame (e.g. back turned)
        adapter = FaceRecognitionAdapter(PROJECT_ROOT / "data" / "faces.json")
        adapter._is_initialized = True
        adapter._app = MagicMock()
        adapter._app.get.return_value = []  # No faces found

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        name, sim, r_stat, _ = adapter.process_event(evt, [(3.8, dummy_frame, 10)], (10, 10, 80, 80))
        evt.recognized_person = name
        evt.recognition_status = r_stat

        self.assertEqual(evt.recognized_person, "Unknown")
        self.assertEqual(evt.recognition_status, RecognitionStatus.NO_FACE)
        self.assertTrue(self.storage.append_event(evt))

    def test_scenario_04_person_carries_waste_without_throwing_it(self):
        """Scenario 4: Person carries waste without throwing it -> No littering event."""
        p1 = TrackedObject("Person_1", 1, "person", 0.92, BoundingBox(100, 100, 200, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "recyclable", 0.88, BoundingBox(140, 220, 170, 270), 1, 0.0)

        # Person carries waste while walking across scene from left to right
        all_events = []
        for i in range(1, 30):
            t = float(i) * 0.2
            shift = i * 10
            p1.bbox = BoundingBox(100 + shift, 100, 200 + shift, 350)
            w1.bbox = BoundingBox(140 + shift, 220, 170 + shift, 270)
            events = self.engine.process_frame([p1], [w1], 640, 480, i, t)
            if events:
                all_events.extend(events)

        self.assertEqual(len(all_events), 0, "Carrying waste must never trigger a littering event")

    def test_scenario_05_person_walks_past_existing_litter(self):
        """Scenario 5: Person walks past existing litter -> No new littering event."""
        # Waste is already stationary on the ground without being held
        w_existing = TrackedObject("Waste_pre", 1, "organic", 0.80, BoundingBox(300, 400, 330, 440), 1, 0.0)

        # Initial frames with no person (identifies as pre-existing litter)
        for i in range(1, 4):
            self.engine.process_frame([], [w_existing], 640, 480, i, float(i) * 0.1)

        traj = self.engine._track_states["Waste_pre"]
        self.assertTrue(traj.is_pre_existing_litter, "Must be tagged as pre-existing litter")

        # Now Person_1 enters and walks past the litter
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(200, 150, 290, 420), 4, 0.4)
        all_events = []
        for i in range(4, 25):
            t = float(i) * 0.2
            # Person walks from left to right past the stationary litter
            p1.bbox = BoundingBox(150 + i * 15, 150, 240 + i * 15, 420)
            events = self.engine.process_frame([p1], [w_existing], 640, 480, i, t)
            if events:
                all_events.extend(events)

        self.assertEqual(len(all_events), 0, "Walking past pre-existing litter must not trigger an event")

    def test_scenario_06_person_throws_bottle(self):
        """Scenario 6: Person throws a bottle -> Bottle remains associated with responsible person."""
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        bottle = TrackedObject("Waste_bottle", 1, "recyclable", 0.89, BoundingBox(140, 220, 170, 270), 1, 0.0)

        # Interacting / holding
        for i in range(1, 4):
            self.engine.process_frame([p1], [bottle], 640, 480, i, float(i) * 0.1)

        # Released and lands
        p1.bbox = BoundingBox(40, 100, 140, 350)
        bottle.bbox = BoundingBox(260, 390, 285, 435)

        confirmed = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            events = self.engine.process_frame([p1], [bottle], 640, 480, f_idx, t)
            if events:
                confirmed.extend(events)

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].waste_category, "recyclable")
        self.assertEqual(confirmed[0].waste_track_id, "Waste_bottle")
        self.assertEqual(confirmed[0].person_track_id, "Person_1")

    def test_scenario_07_person_throws_plastic_bag(self):
        """Scenario 7: Person throws plastic bag -> Bag remains detectable and trackable."""
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        bag = TrackedObject("Waste_bag", 1, "residual", 0.82, BoundingBox(140, 220, 180, 280), 1, 0.0)

        for i in range(1, 4):
            self.engine.process_frame([p1], [bag], 640, 480, i, float(i) * 0.1)

        p1.bbox = BoundingBox(20, 100, 120, 350)
        bag.bbox = BoundingBox(310, 410, 360, 460)

        confirmed = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            events = self.engine.process_frame([p1], [bag], 640, 480, f_idx, t)
            if events:
                confirmed.extend(events)

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].waste_category, "residual")
        self.assertEqual(confirmed[0].person_track_id, "Person_1")

    def test_scenario_08_two_people_nearby_actual_thrower_vs_ambiguous(self):
        """Scenario 8: Two people nearby -> Associate actual thrower or mark unattributed/ambiguous."""
        # Case 8a: Person_1 clearly held waste; Person_2 is standing 150px away
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        p2 = TrackedObject("Person_2", 2, "person", 0.88, BoundingBox(350, 100, 450, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "recyclable", 0.85, BoundingBox(140, 220, 170, 270), 1, 0.0)

        for i in range(1, 5):
            self.engine.process_frame([p1, p2], [w1], 640, 480, i, float(i) * 0.1)

        p1.bbox = BoundingBox(50, 100, 150, 350)
        w1.bbox = BoundingBox(220, 390, 250, 430)

        confirmed = []
        for f_idx, t in enumerate([0.6, 1.6, 2.6, 3.9], start=5):
            events = self.engine.process_frame([p1, p2], [w1], 640, 480, f_idx, t)
            if events:
                confirmed.extend(events)

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].person_track_id, "Person_1", "Must attribute actual thrower")
        self.assertEqual(confirmed[0].attribution_status, AttributionStatus.ATTRIBUTED)

        # Case 8b: Ambiguous interaction where both people were equally close without clear holding
        engine2 = PersonWasteAssociationEngine(zone_manager=self.zones)
        pA = TrackedObject("Person_A", 1, "person", 0.85, BoundingBox(100, 200, 180, 400), 1, 0.0)
        pB = TrackedObject("Person_B", 2, "person", 0.85, BoundingBox(220, 200, 300, 400), 1, 0.0)
        w_amb = TrackedObject("Waste_amb", 2, "recyclable", 0.80, BoundingBox(190, 300, 210, 330), 1, 0.0)

        for i in range(1, 4):
            engine2.process_frame([pA, pB], [w_amb], 640, 480, i, float(i) * 0.1)

        w_amb.bbox = BoundingBox(200, 400, 220, 430)
        confirmed_amb = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            events = engine2.process_frame([pA, pB], [w_amb], 640, 480, f_idx, t)
            if events:
                confirmed_amb.extend(events)

        self.assertEqual(len(confirmed_amb), 1)
        self.assertEqual(confirmed_amb[0].attribution_status, AttributionStatus.AMBIGUOUS)
        self.assertIsNone(confirmed_amb[0].person_track_id)

    def test_scenario_09_waste_moves_or_rolls_after_impact_avoids_premature_confirmation(self):
        """Scenario 9: Waste moves or rolls after impact -> Avoid premature confirmation."""
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        bottle = TrackedObject("Waste_roll", 1, "recyclable", 0.90, BoundingBox(140, 220, 170, 270), 1, 0.0)

        for i in range(1, 4):
            self.engine.process_frame([p1], [bottle], 640, 480, i, float(i) * 0.1)

        p1.bbox = BoundingBox(50, 100, 150, 350)

        # Bottle lands at t=0.5 and rolls continuously from x=200 to x=350 at speed > 40 px/s
        rolling_events = []
        for i, t in enumerate([0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9], start=4):
            # Bottle moves 18px per step (speed = 18 / 0.3 = 60 px/s > 25 px/s threshold)
            bottle.bbox = BoundingBox(200 + i * 18, 400, 230 + i * 18, 440)
            events = self.engine.process_frame([p1], [bottle], 640, 480, i, t)
            if events:
                rolling_events.extend(events)

        self.assertEqual(len(rolling_events), 0, "Rolling bottle must NOT prematurely confirm littering")

    def test_scenario_10_waste_remains_stationary_for_3_to_5_seconds(self):
        """Scenario 10: Waste remains stationary for 3-5 seconds -> Confirm event."""
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        bottle = TrackedObject("Waste_1", 1, "recyclable", 0.90, BoundingBox(140, 220, 170, 270), 1, 0.0)

        for i in range(1, 4):
            self.engine.process_frame([p1], [bottle], 640, 480, i, float(i) * 0.1)

        p1.bbox = BoundingBox(40, 100, 140, 350)
        # Bottle lands and becomes stationary at (280, 400)
        bottle.bbox = BoundingBox(280, 400, 310, 440)

        events_at_2s = self.engine.process_frame([p1], [bottle], 640, 480, 4, 0.5)
        self.assertEqual(len(events_at_2s), 0)

        events_at_2_5s = self.engine.process_frame([p1], [bottle], 640, 480, 5, 2.5)
        self.assertEqual(len(events_at_2_5s), 0, "Not yet stationary for 3.0s")

        events_at_3_6s = self.engine.process_frame([p1], [bottle], 640, 480, 6, 3.6)
        self.assertEqual(len(events_at_3_6s), 1, "Confirmed after >= 3.0s stationary duration")
        self.assertEqual(events_at_3_6s[0].event_type, DisposalEventType.OPEN_GROUND_LITTERING)

    def test_scenario_11_person_throws_waste_outside_bin(self):
        """Scenario 11: Person throws waste outside a bin -> Open-area littering event."""
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "recyclable", 0.85, BoundingBox(140, 220, 170, 270), 1, 0.0)

        for i in range(1, 4):
            self.engine.process_frame([p1], [w1], 640, 480, i, float(i) * 0.1)

        p1.bbox = BoundingBox(40, 100, 140, 350)
        # Dropped in ground zone at x=200 (far from bin which is at x=[0.70..0.95] * 640 = [448..608])
        w1.bbox = BoundingBox(200, 380, 230, 420)

        confirmed = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            events = self.engine.process_frame([p1], [w1], 640, 480, f_idx, t)
            if events:
                confirmed.extend(events)

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].event_type, DisposalEventType.OPEN_GROUND_LITTERING)
        self.assertEqual(confirmed[0].ground_zone_id, "open_ground_default")

    def test_scenario_12_person_throws_waste_into_bin(self):
        """Scenario 12: Person throws waste into a bin -> Bin-disposal event."""
        # Frame 1: Person holds waste above the bin (y2 = 220, outside bin which starts at y=264)
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(450, 150, 550, 400), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "recyclable", 0.90, BoundingBox(480, 180, 510, 220), 1, 0.0)

        events_f1 = self.engine.process_frame([p1], [w1], 640, 480, 1, 0.1)
        self.assertEqual(len(events_f1), 0, "No event before entering bin")

        # Frame 2: Waste enters bin polygon: bin_primary is at [0.70..0.95]*640 = [448..608], [0.55..0.95]*480 = [264..456]
        w1.bbox = BoundingBox(490, 300, 520, 340)

        events = self.engine.process_frame([p1], [w1], 640, 480, 2, 0.2)
        self.assertEqual(len(events), 1, "Must confirm bin disposal immediately upon entering bin")
        self.assertEqual(events[0].event_type, DisposalEventType.BIN_DISPOSAL)
        self.assertEqual(events[0].ground_zone_id, "bin_primary")
        self.assertEqual(events[0].person_track_id, "Person_1")

    def test_scenario_13_reject_false_human_candidate_detection(self):
        """Scenario 13: Human hand or clothing is detected as candidate waste -> Reject false detection."""
        detector = WasteDetector(PROJECT_ROOT / "models" / "waste" / "waste_detection_yolo11m.pt", verify_sha256=False)
        person_box = BoundingBox(100, 100, 250, 450)

        # False detection 1: Matching entire body
        full_body = BoundingBox(105, 105, 245, 445)
        self.assertTrue(detector._is_false_human_detection(full_body, 0.70, [person_box]))

        # False detection 2: Torso/clothing patch with moderate confidence
        torso_patch = BoundingBox(130, 180, 210, 290)
        self.assertTrue(detector._is_false_human_detection(torso_patch, 0.50, [person_box]))

        # Genuine held waste extending beyond hand
        genuine_item = BoundingBox(230, 280, 275, 330)
        self.assertFalse(detector._is_false_human_detection(genuine_item, 0.85, [person_box]))

    def test_scenario_14_multiple_littering_events_record_separately_without_duplicate_ids(self):
        """Scenario 14: Multiple littering events occur -> Record separate event IDs without duplication."""
        # Event 1 by Person_1
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(50, 100, 150, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "organic", 0.88, BoundingBox(90, 220, 120, 270), 1, 0.0)

        for i in range(1, 4):
            self.engine.process_frame([p1], [w1], 640, 480, i, float(i) * 0.1)

        w1.bbox = BoundingBox(120, 390, 150, 430)
        e1 = None
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            evs = self.engine.process_frame([p1], [w1], 640, 480, f_idx, t)
            if evs:
                e1 = evs[0]

        self.assertIsNotNone(e1)
        self.storage.append_event(e1)

        # Event 2 by Person_2 after cooldown (t=10.0s)
        p2 = TrackedObject("Person_2", 2, "person", 0.90, BoundingBox(300, 100, 400, 350), 20, 10.0)
        w2 = TrackedObject("Waste_2", 2, "recyclable", 0.85, BoundingBox(340, 220, 370, 270), 20, 10.0)

        for i in range(21, 24):
            self.engine.process_frame([p2], [w2], 640, 480, i, 10.0 + (i - 20) * 0.1)

        w2.bbox = BoundingBox(380, 390, 410, 430)
        e2 = None
        for f_idx, t in enumerate([10.5, 11.5, 12.5, 13.8], start=24):
            evs = self.engine.process_frame([p2], [w2], 640, 480, f_idx, t)
            if evs:
                e2 = evs[0]

        self.assertIsNotNone(e2)
        self.assertNotEqual(e1.event_id, e2.event_id, "Event IDs must be distinct")
        self.storage.append_event(e2)
        self.assertEqual(self.storage.total_records, 2, "Both events must be stored in CSV")

    def test_scenario_15_evidence_recording_failure_handles_gracefully_without_fabrication(self):
        """Scenario 15: Evidence recording is unavailable -> Report failure; do not fabricate a video path."""
        buffer = RollingFrameBuffer(max_seconds=5.0)
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        buffer.append(10.0, dummy_frame, 1)

        # Use invalid directory to simulate failure
        invalid_dir = Path("Z:/non_existent_drive_12345/evidence")
        recorder = EvidenceRecorder(evidence_dir=invalid_dir, frame_buffer=buffer)

        event = ConfirmedLitterEvent(
            event_id="evt_fail_test",
            event_type=DisposalEventType.OPEN_GROUND_LITTERING,
            estimated_release_time="2026-11-18T10:00:00Z",
            estimated_release_timestamp=10.0,
            confirmed_at="2026-11-18T10:00:05Z",
            confirmed_timestamp=15.0,
            waste_category="recyclable",
            waste_confidence=0.90,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
        )

        # Attempt to save evidence
        recorder._save_evidence(event, face_crop=dummy_frame)

        # Path must NOT be fabricated!
        self.assertEqual(event.video_evidence_path, "", "Must not fabricate a video path on failure")
        self.assertTrue("failed" in event.notes.lower() or "error" in event.notes.lower(), "Must report failure in notes")

        # CSV storage still appends the event with empty evidence path and failure notes
        self.assertTrue(self.storage.append_event(event))
        self.assertEqual(self.storage.total_records, 1)


if __name__ == "__main__":
    unittest.main()
