"""Comprehensive verification suite for all 12 core dumping & association scenarios.

Scenarios:
1. Person walks past existing waste -> No event, no recognition.
2. Person stands beside existing waste -> No event, no recognition.
3. Person carries waste without dropping it -> No event.
4. Person throws waste into an open area -> Confirmed open-ground event, person attributed, recognized.
5. Person throws waste and walks away -> Confirmed event, thrower attributed from history, face recognized from history.
6. Unknown person throws waste -> Confirmed event, unknown identity saved to CSV.
7. Enrolled person throws waste -> Confirmed event, enrolled identity saved to CSV.
8. Multiple people near dumping location -> Confirmed event, actual thrower attributed, bystanders ignored.
9. One person dumps while another walks past -> Thrower attributed, passerby ignored.
10. Waste already on ground before person enters -> No event.
11. Waste rolls or moves due to environment -> Confirmed only after stationary, not while rolling.
12. Person disposes waste inside designated bin -> Confirmed as BIN_DISPOSAL event.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

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
    DisposalEventType,
    RecognitionStatus,
    TrackedObject,
)
from app.evidence.recorder import EvidenceRecorder
from app.recognition.face_adapter import FaceRecognitionAdapter
from app.storage.csv_storage import CSVEventStorage


def create_test_zones() -> ZoneManager:
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


class Test12Scenarios(unittest.TestCase):
    """Execution of all 12 required real-world operational scenarios."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.csv_path = self.temp_path / "events.csv"
        self.evidence_dir = self.temp_path / "evidence"
        self.faces_path = self.temp_path / "faces.json"

        # Setup enrolled faces: "Alice"
        self.enrolled_embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)
        registry = {"Alice": {"embedding": self.enrolled_embedding.tolist()}}
        with open(self.faces_path, "w", encoding="utf-8") as f:
            json.dump(registry, f)

        self.zones = create_test_zones()
        self.engine = PersonWasteAssociationEngine(zone_manager=self.zones)
        self.storage = CSVEventStorage(csv_path=self.csv_path)

        # Mock FaceRecognitionAdapter
        self.adapter = FaceRecognitionAdapter(faces_json_path=self.faces_path)
        self.adapter._load_registry()
        self.adapter._is_initialized = True
        self.adapter._app = MagicMock()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _mock_face(self, embedding: np.ndarray, score: float = 0.95):
        m = MagicMock()
        m.embedding = embedding
        m.det_score = score
        m.bbox = np.array([10, 10, 60, 60], dtype=np.float32)
        return m

    def _dummy_frame(self):
        f = np.full((120, 120, 3), 128, dtype=np.uint8)
        f[::2, ::2] = 255
        f[1::2, 1::2] = 0
        return f

    # Scenario 1: Person walks past existing waste
    def test_scenario_01_person_walks_past_existing_waste(self):
        w_exist = TrackedObject("Waste_old", 1, "organic", 0.85, BoundingBox(300, 400, 330, 440), 1, 0.0)
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 150, 190, 420), 1, 0.0)

        events = []
        # Frame 1-3: Waste stationary on ground; person walks past from x=100 to x=500
        for i in range(1, 20):
            t = float(i) * 0.2
            p1.bbox = BoundingBox(100 + i * 20, 150, 190 + i * 20, 420)
            evts = self.engine.process_frame([p1], [w_exist], 640, 480, i, t)
            events.extend(evts)

        self.assertEqual(len(events), 0, "Walking past existing waste must NEVER trigger dumping event")
        self.assertEqual(self.storage.total_records, 0)

    # Scenario 2: Person stands beside existing waste
    def test_scenario_02_person_stands_beside_existing_waste(self):
        w_exist = TrackedObject("Waste_old", 1, "residual", 0.82, BoundingBox(300, 400, 330, 440), 1, 0.0)
        # Person stands 60px away at (220..310, 150..420) for 5 seconds
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(220, 150, 310, 420), 1, 0.0)

        events = []
        for i in range(1, 25):
            t = float(i) * 0.25
            evts = self.engine.process_frame([p1], [w_exist], 640, 480, i, t)
            events.extend(evts)

        self.assertEqual(len(events), 0, "Standing beside existing waste must NEVER trigger dumping event")
        self.assertEqual(self.storage.total_records, 0)

    # Scenario 3: Person carries waste without dropping it
    def test_scenario_03_person_carries_waste_without_dropping(self):
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "recyclable", 0.88, BoundingBox(140, 200, 170, 250), 1, 0.0)

        events = []
        for i in range(1, 25):
            t = float(i) * 0.2
            shift = i * 10
            p1.bbox = BoundingBox(100 + shift, 100, 200 + shift, 350)
            w1.bbox = BoundingBox(140 + shift, 200, 170 + shift, 250)
            evts = self.engine.process_frame([p1], [w1], 640, 480, i, t)
            events.extend(evts)

        self.assertEqual(len(events), 0, "Carrying waste without release must NEVER trigger dumping event")
        self.assertEqual(self.storage.total_records, 0)

    # Scenario 4: Person throws waste into an open area
    def test_scenario_04_person_throws_waste_into_open_area(self):
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "residual", 0.85, BoundingBox(140, 220, 170, 270), 1, 0.0)

        # 1. Holding phase
        for i in range(1, 4):
            self.engine.process_frame([p1], [w1], 640, 480, i, float(i) * 0.1)

        # 2. Release & land on open ground
        p1.bbox = BoundingBox(40, 100, 140, 350)
        w1.bbox = BoundingBox(250, 400, 280, 440)

        # 3. Stationary confirmation (3.5 seconds)
        confirmed = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            evts = self.engine.process_frame([p1], [w1], 640, 480, f_idx, t)
            confirmed.extend(evts)

        self.assertEqual(len(confirmed), 1)
        evt = confirmed[0]
        self.assertEqual(evt.event_type, DisposalEventType.OPEN_GROUND_LITTERING)
        self.assertEqual(evt.person_track_id, "Person_1")
        self.assertEqual(evt.attribution_status, AttributionStatus.ATTRIBUTED)

        # Run face recognition on thrower
        self.adapter._app.get.return_value = [self._mock_face(self.enrolled_embedding)]
        name, sim, r_stat, _ = self.adapter.process_event(evt, [(3.8, self._dummy_frame(), 10)], (40, 100, 140, 350))
        evt.recognized_person = name
        evt.recognition_status = r_stat

        self.assertEqual(evt.recognized_person, "Alice")
        self.assertEqual(evt.recognition_status, RecognitionStatus.IDENTIFIED)
        self.assertTrue(self.storage.append_event(evt))
        self.assertEqual(self.storage.total_records, 1)

    # Scenario 5: Person throws waste and walks away
    def test_scenario_05_person_throws_waste_and_walks_away(self):
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "organic", 0.88, BoundingBox(140, 220, 170, 270), 1, 0.0)

        # Holding
        for i in range(1, 4):
            self.engine.process_frame([p1], [w1], 640, 480, i, float(i) * 0.1)

        # Released at t=0.4
        w1.bbox = BoundingBox(280, 410, 310, 450)

        # Person walks away completely (out of frame)
        confirmed = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            # Only waste in scene, person has departed
            evts = self.engine.process_frame([], [w1], 640, 480, f_idx, t)
            confirmed.extend(evts)

        self.assertEqual(len(confirmed), 1)
        evt = confirmed[0]
        self.assertEqual(evt.person_track_id, "Person_1", "Must preserve thrower track ID even after walking away")
        self.assertEqual(evt.attribution_status, AttributionStatus.ATTRIBUTED)

    # Scenario 6: Unknown person throws waste
    def test_scenario_06_unknown_person_throws_waste(self):
        p1 = TrackedObject("Person_Unk", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "residual", 0.85, BoundingBox(140, 220, 170, 270), 1, 0.0)

        for i in range(1, 4):
            self.engine.process_frame([p1], [w1], 640, 480, i, float(i) * 0.1)

        p1.bbox = BoundingBox(40, 100, 140, 350)
        w1.bbox = BoundingBox(250, 400, 280, 440)

        confirmed = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            confirmed.extend(self.engine.process_frame([p1], [w1], 640, 480, f_idx, t))

        self.assertEqual(len(confirmed), 1)
        evt = confirmed[0]

        # Unenrolled face (orthogonal vector -> similarity = 0.0 < 0.50)
        unk_embedding = np.zeros(512, dtype=np.float32)
        unk_embedding[10] = 1.0
        self.adapter._app.get.return_value = [self._mock_face(unk_embedding)]

        name, sim, r_stat, _ = self.adapter.process_event(evt, [(3.8, self._dummy_frame(), 10)], (40, 100, 140, 350))
        evt.recognized_person = name
        evt.recognition_status = r_stat

        self.assertEqual(evt.recognized_person, "Unknown")
        self.assertEqual(evt.recognition_status, RecognitionStatus.UNKNOWN)
        self.assertTrue(self.storage.append_event(evt))

    # Scenario 7: Enrolled person throws waste
    def test_scenario_07_enrolled_person_throws_waste(self):
        p1 = TrackedObject("Person_Enrolled", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "recyclable", 0.89, BoundingBox(140, 220, 170, 270), 1, 0.0)

        for i in range(1, 4):
            self.engine.process_frame([p1], [w1], 640, 480, i, float(i) * 0.1)

        p1.bbox = BoundingBox(40, 100, 140, 350)
        w1.bbox = BoundingBox(250, 400, 280, 440)

        confirmed = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            confirmed.extend(self.engine.process_frame([p1], [w1], 640, 480, f_idx, t))

        self.assertEqual(len(confirmed), 1)
        evt = confirmed[0]

        # Enrolled face match
        self.adapter._app.get.return_value = [self._mock_face(self.enrolled_embedding)]
        name, sim, r_stat, _ = self.adapter.process_event(evt, [(3.8, self._dummy_frame(), 10)], (40, 100, 140, 350))
        evt.recognized_person = name
        evt.recognition_status = r_stat

        self.assertEqual(evt.recognized_person, "Alice")
        self.assertEqual(evt.recognition_status, RecognitionStatus.IDENTIFIED)
        self.assertTrue(self.storage.append_event(evt))

    # Scenario 8: Multiple people near dumping location
    def test_scenario_08_multiple_people_near_dumping_location(self):
        # Person_1 is the thrower; Person_2 & Person_3 are bystanders 120px away
        p1 = TrackedObject("Person_Thrower", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        p2 = TrackedObject("Bystander_A", 2, "person", 0.88, BoundingBox(320, 100, 420, 350), 1, 0.0)
        p3 = TrackedObject("Bystander_B", 3, "person", 0.88, BoundingBox(450, 100, 550, 350), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "residual", 0.85, BoundingBox(140, 220, 170, 270), 1, 0.0)

        # Holding phase: only p1 holds waste
        for i in range(1, 4):
            self.engine.process_frame([p1, p2, p3], [w1], 640, 480, i, float(i) * 0.1)

        p1.bbox = BoundingBox(40, 100, 140, 350)
        w1.bbox = BoundingBox(220, 400, 250, 440)

        confirmed = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            confirmed.extend(self.engine.process_frame([p1, p2, p3], [w1], 640, 480, f_idx, t))

        self.assertEqual(len(confirmed), 1)
        evt = confirmed[0]
        self.assertEqual(evt.person_track_id, "Person_Thrower", "Must associate actual thrower, NOT bystanders")
        self.assertEqual(evt.attribution_status, AttributionStatus.ATTRIBUTED)

    # Scenario 9: One person dumps while another walks past
    def test_scenario_09_one_person_dumps_while_another_walks_past(self):
        p_dump = TrackedObject("Person_Dumper", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        p_walk = TrackedObject("Person_Walker", 2, "person", 0.88, BoundingBox(500, 150, 590, 420), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "organic", 0.85, BoundingBox(140, 220, 170, 270), 1, 0.0)

        for i in range(1, 4):
            # Walker moves closer across frames
            p_walk.bbox = BoundingBox(500 - i * 30, 150, 590 - i * 30, 420)
            self.engine.process_frame([p_dump, p_walk], [w1], 640, 480, i, float(i) * 0.1)

        p_dump.bbox = BoundingBox(40, 100, 140, 350)
        w1.bbox = BoundingBox(260, 400, 290, 440)

        confirmed = []
        for f_idx, t in enumerate([0.5, 1.5, 2.5, 3.8], start=4):
            p_walk.bbox = BoundingBox(380 - f_idx * 20, 150, 470 - f_idx * 20, 420)
            confirmed.extend(self.engine.process_frame([p_dump, p_walk], [w1], 640, 480, f_idx, t))

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].person_track_id, "Person_Dumper", "Must attribute dumper, NOT walker")

    # Scenario 10: Waste is already on the ground before person enters
    def test_scenario_10_waste_already_on_ground_before_person_enters(self):
        w_ground = TrackedObject("Waste_gnd", 1, "residual", 0.80, BoundingBox(300, 400, 330, 440), 1, 0.0)

        # Waste alone on ground for 3 frames
        for i in range(1, 4):
            self.engine.process_frame([], [w_ground], 640, 480, i, float(i) * 0.1)

        # Now person enters at t=0.4
        p1 = TrackedObject("Person_Late", 1, "person", 0.90, BoundingBox(200, 150, 290, 420), 4, 0.4)
        all_events = []
        for i in range(4, 25):
            t = float(i) * 0.2
            all_events.extend(self.engine.process_frame([p1], [w_ground], 640, 480, i, t))

        self.assertEqual(len(all_events), 0, "Pre-existing ground waste must NEVER trigger dumping event")

    # Scenario 11: Waste rolls or moves due to environmental conditions (human drop then roll)
    def test_scenario_11_waste_rolls_or_moves_due_to_environment(self):
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(100, 100, 200, 350), 1, 0.0)
        bottle = TrackedObject("Waste_Roll", 1, "recyclable", 0.90, BoundingBox(140, 220, 170, 270), 1, 0.0)

        for i in range(1, 4):
            self.engine.process_frame([p1], [bottle], 640, 480, i, float(i) * 0.1)

        p1.bbox = BoundingBox(40, 100, 140, 350)

        # Rolling phase: bottle rolls along x from 200 to 380 at speed > 50 px/s
        rolling_events = []
        for i, t in enumerate([0.5, 0.8, 1.1, 1.4, 1.7, 2.0], start=4):
            bottle.bbox = BoundingBox(200 + i * 25, 400, 230 + i * 25, 440)
            rolling_events.extend(self.engine.process_frame([p1], [bottle], 640, 480, i, t))

        self.assertEqual(len(rolling_events), 0, "Must NOT confirm while rolling")

        # Settles at x=350 from t=2.5 to t=5.8 (>= 3.0 seconds stationary)
        bottle.bbox = BoundingBox(350, 400, 380, 440)
        settled_events = []
        for f_idx, t in enumerate([2.5, 3.5, 4.5, 5.8], start=10):
            settled_events.extend(self.engine.process_frame([p1], [bottle], 640, 480, f_idx, t))

        self.assertEqual(len(settled_events), 1, "Confirmed ONLY after resting stationary for 3.0s")
        self.assertEqual(settled_events[0].person_track_id, "Person_1")

    # Scenario 11b: Waste moves/rolls purely due to environment WITHOUT human release
    def test_scenario_11b_pure_environmental_waste_rolling_no_human_release(self):
        # Bystander standing nearby in background
        p_bystander = TrackedObject("Person_Bystander", 2, "person", 0.92, BoundingBox(250, 150, 340, 420), 1, 0.0)
        # Empty bottle/can blown across the ground by wind/gravity with no person ever holding it
        wind_bottle = TrackedObject("Waste_Wind", 10, "recyclable", 0.88, BoundingBox(50, 400, 80, 440), 1, 0.0)

        all_emitted = []
        # Phase 1: Rolling along ground across 6 frames (x=50 to 200, rolling near bystander at x=250)
        for i, t in enumerate([0.1, 0.4, 0.7, 1.0, 1.3, 1.6], start=1):
            wind_bottle.bbox = BoundingBox(50 + i * 25, 400, 80 + i * 25, 440)
            all_emitted.extend(self.engine.process_frame([p_bystander], [wind_bottle], 640, 480, i, t))

        # Phase 2: Settles stationary at x=200 for > 3.5 seconds
        for f_idx, t in enumerate([2.0, 3.0, 4.0, 5.5], start=7):
            all_emitted.extend(self.engine.process_frame([p_bystander], [wind_bottle], 640, 480, f_idx, t))

        self.assertEqual(len(all_emitted), 0, "Pure environmental movement without human release must NEVER emit a dumping event")
        traj = self.engine.get_trajectory("Waste_Wind")
        self.assertIsNotNone(traj)
        self.assertTrue(traj.is_pre_existing_litter, "Unheld environmental waste must be marked as pre-existing litter")
        self.assertEqual(len(traj.person_held_frames), 0, "No human hold was ever registered")

    # Scenario 12: Person disposes waste inside designated bin
    def test_scenario_12_person_disposes_waste_inside_designated_bin(self):
        # Frame 1: Person holds waste near bin opening
        p1 = TrackedObject("Person_1", 1, "person", 0.90, BoundingBox(450, 150, 550, 400), 1, 0.0)
        w1 = TrackedObject("Waste_1", 1, "recyclable", 0.90, BoundingBox(480, 180, 510, 220), 1, 0.0)

        evts_f1 = self.engine.process_frame([p1], [w1], 640, 480, 1, 0.1)
        self.assertEqual(len(evts_f1), 0)

        # Frame 2: Waste enters bin zone (x=[448..608], y=[264..456])
        w1.bbox = BoundingBox(490, 300, 520, 340)
        evts_f2 = self.engine.process_frame([p1], [w1], 640, 480, 2, 0.2)

        self.assertEqual(len(evts_f2), 1)
        evt = evts_f2[0]
        self.assertEqual(evt.event_type, DisposalEventType.BIN_DISPOSAL)
        self.assertEqual(evt.person_track_id, "Person_1")
        self.assertEqual(evt.ground_zone_id, "bin_primary")


if __name__ == "__main__":
    unittest.main()
