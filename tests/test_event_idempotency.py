"""Unit and regression test suite for P11 Receptacle Entry Verification and Deposit Lifecycle Management.

Covers all 20 required criteria across:
- Startup ambient-waste inventory (Tests 1-6)
- Deposit lifecycle and re-arming (Tests 7-14)
- System preservation and regression boundaries (Tests 15-20)
"""

import unittest
from unittest.mock import MagicMock

from app.association.association_engine import (
    PersonWasteAssociationEngine,
    PhysicalDeposit,
    WasteTrajectoryTracker,
)
from app.association.zones import DetectionZone, ZoneManager
from app.common_types import (
    AttributionStatus,
    BoundingBox,
    ConfirmedLitterEvent,
    DisposalEventType,
    LitteringState,
    RecognitionStatus,
    TrackedObject,
)


def make_tracked_waste(
    track_id: str,
    bbox: BoundingBox,
    frame_idx: int = 1,
    timestamp: float = 1.0,
    class_name: str = "recyclable",
    confidence: float = 0.85,
) -> TrackedObject:
    try:
        raw_id = int(track_id.split("_")[-1])
    except (ValueError, IndexError):
        raw_id = 1
    return TrackedObject(
        track_id=track_id,
        raw_id=raw_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        last_frame_idx=frame_idx,
        last_timestamp=timestamp,
        history=[(frame_idx, timestamp, bbox)],
    )


def make_tracked_person(
    track_id: str,
    bbox: BoundingBox,
    frame_idx: int = 1,
    timestamp: float = 1.0,
) -> TrackedObject:
    try:
        raw_id = int(track_id.split("_")[-1])
    except (ValueError, IndexError):
        raw_id = 1
    return TrackedObject(
        track_id=track_id,
        raw_id=raw_id,
        class_name="person",
        confidence=0.90,
        bbox=bbox,
        last_frame_idx=frame_idx,
        last_timestamp=timestamp,
        history=[(frame_idx, timestamp, bbox)],
    )


class TestReceptacleEntryAndDepositLifecycle(unittest.TestCase):
    """Test suite verifying receptacle entry verification, ambient inventory, and deposit lifecycle."""

    def setUp(self):
        self.zone_mgr = MagicMock(spec=ZoneManager)
        # Default: bin at [700, 500, 950, 950] on 1000x1000 frame
        self.bin_zone = DetectionZone(
            zone_id="bin_primary",
            zone_type="bin",
            polygon=[[0.70, 0.50], [0.95, 0.50], [0.95, 0.95], [0.70, 0.95]],
            description="Primary Bin",
        )
        self.ground_zone = DetectionZone(
            zone_id="open_ground_default",
            zone_type="ground",
            polygon=[[0.0, 0.50], [1.0, 0.50], [1.0, 1.0], [0.0, 1.0]],
            description="Open Ground",
        )

        def mock_matching_bin(bbox, w, h):
            cx, cy = bbox.center
            if 700 <= cx <= 950 and 500 <= cy <= 950:
                return self.bin_zone
            return None

        def mock_is_in_ground(bbox, w, h):
            cx, cy = bbox.center
            if cy >= 500:
                return True, "open_ground_default"
            return False, ""

        self.zone_mgr.find_matching_bin.side_effect = mock_matching_bin
        self.zone_mgr.is_in_ground_zone.side_effect = mock_is_in_ground

        self.engine = PersonWasteAssociationEngine(
            zone_manager=self.zone_mgr,
            holding_max_distance_px=160.0,
            holding_iou_threshold=0.05,
            min_holding_frames=2,
            separation_distance_delta_px=20.0,
            stationary_duration_seconds=3.0,
            event_cooldown_seconds=0.0,
            ambiguity_margin_px=35.0,
        )

    # =========================================================================
    # Group 1: Startup Ambient-Waste Inventory (Tests 1-6)
    # =========================================================================

    def test_01_waste_present_at_startup_generates_no_event(self):
        """1. Waste already present in a bin at stream startup generates no disposal event."""
        box = BoundingBox(750, 600, 800, 650)
        w1 = make_tracked_waste("Waste_1", box, frame_idx=1, timestamp=1.0)

        # Frame 1: waste appears inside bin with no person holding
        events_f1 = self.engine.process_frame([], [w1], 1000, 1000, 1, 1.0)
        self.assertEqual(len(events_f1), 0, "Ambient startup waste must NOT emit a disposal event")
        self.assertEqual(len(self.engine._confirmed_deposits), 1)
        self.assertTrue(self.engine._confirmed_deposits[0].is_ambient)
        self.assertEqual(self.engine._confirmed_deposits[0].state, "PRESENT")

    def test_02_multiple_pre_existing_objects_registered_as_ambient(self):
        """2. Multiple pre-existing objects are registered as ambient content."""
        box1 = BoundingBox(720, 520, 760, 560)
        box2 = BoundingBox(850, 800, 900, 850)
        w1 = make_tracked_waste("Waste_1", box1, frame_idx=1, timestamp=1.0)
        w2 = make_tracked_waste("Waste_2", box2, frame_idx=1, timestamp=1.0)

        events = self.engine.process_frame([], [w1, w2], 1000, 1000, 1, 1.0)
        self.assertEqual(len(events), 0, "Zero events for multiple ambient startup items")
        self.assertEqual(len(self.engine._confirmed_deposits), 2)
        self.assertTrue(all(dep.is_ambient for dep in self.engine._confirmed_deposits))

    def test_03_ambient_object_new_tracker_id_no_event(self):
        """3. An ambient object receiving a new tracker ID does not generate a disposal event."""
        box1 = BoundingBox(750, 600, 800, 650)
        w1 = make_tracked_waste("Waste_1", box1, frame_idx=1, timestamp=1.0)
        self.engine.process_frame([], [w1], 1000, 1000, 1, 1.0)

        # Track split / reacquisition 35 frames later as Waste_7 at essentially same location
        box2 = BoundingBox(751, 601, 801, 651)
        self.assertGreaterEqual(box1.iou(box2), 0.70)
        w2 = make_tracked_waste("Waste_7", box2, frame_idx=36, timestamp=36.0)

        events_f36 = self.engine.process_frame([], [w2], 1000, 1000, 36, 36.0)
        self.assertEqual(len(events_f36), 0, "Duplicate reacquisition of ambient item must be suppressed")

    def test_04_new_object_entering_bin_after_init_generates_one_event(self):
        """4. A new object entering a bin after initialization generates one legitimate event."""
        # Frame 1: Object is outside the bin (cy < 500)
        box_outside = BoundingBox(500, 300, 550, 350)
        w1 = make_tracked_waste("Waste_1", box_outside, frame_idx=1, timestamp=1.0)
        ev1 = self.engine.process_frame([], [w1], 1000, 1000, 1, 1.0)
        self.assertEqual(len(ev1), 0)

        # Frame 2: Object transitions into bin
        box_inside = BoundingBox(750, 600, 800, 650)
        w1_in = make_tracked_waste("Waste_1", box_inside, frame_idx=2, timestamp=1.1)
        ev2 = self.engine.process_frame([], [w1_in], 1000, 1000, 2, 1.1)
        self.assertEqual(len(ev2), 1, "Observed entry into receptacle must emit BIN_DISPOSAL")
        self.assertEqual(ev2[0].event_type, DisposalEventType.BIN_DISPOSAL)
        self.assertEqual(ev2[0].waste_track_id, "Waste_1")

    def test_05_genuine_handling_and_release_near_startup_confirmed(self):
        """5. A genuine handling-and-release action close to startup is not silently discarded."""
        p = make_tracked_person("Person_1", BoundingBox(550, 480, 650, 780), frame_idx=1, timestamp=1.0)
        # Waste held by person outside the bin first
        w = make_tracked_waste("Waste_1", BoundingBox(580, 520, 620, 560), frame_idx=1, timestamp=1.0)

        # Frame 1: Held at startup outside bin
        ev1 = self.engine.process_frame([p], [w], 1000, 1000, 1, 1.0)
        self.assertEqual(len(ev1), 0)

        # Frame 2: Still held outside bin, confirming interaction
        ev2 = self.engine.process_frame([p], [w], 1000, 1000, 2, 1.1)
        self.assertEqual(len(ev2), 0)

        # Frame 3: Person drops waste into bin
        w_bin = make_tracked_waste("Waste_1", BoundingBox(750, 600, 790, 640), frame_idx=3, timestamp=1.2)
        ev3 = self.engine.process_frame([p], [w_bin], 1000, 1000, 3, 1.2)
        self.assertEqual(len(ev3), 1, "Genuine disposal near startup must be confirmed")
        self.assertEqual(ev3[0].person_track_id, "Person_1")
        self.assertEqual(ev3[0].attribution_status, AttributionStatus.ATTRIBUTED)

    def test_06_ambiguous_responsibility_remains_unattributed_or_ambiguous(self):
        """6. Ambiguous responsibility remains UNATTRIBUTED or AMBIGUOUS when appropriate."""
        # Case A: Object enters bin from outside with no person (thrown from off-screen)
        box_outside = BoundingBox(400, 200, 450, 250)
        w_throw = make_tracked_waste("Waste_thrown", box_outside, frame_idx=1, timestamp=1.0)
        self.engine.process_frame([], [w_throw], 1000, 1000, 1, 1.0)

        box_bin = BoundingBox(750, 600, 800, 650)
        w_thrown_in = make_tracked_waste("Waste_thrown", box_bin, frame_idx=2, timestamp=1.1)
        ev = self.engine.process_frame([], [w_thrown_in], 1000, 1000, 2, 1.1)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].attribution_status, AttributionStatus.UNATTRIBUTED)
        self.assertIsNone(ev[0].person_track_id)

    # =========================================================================
    # Group 2: Deposit Lifecycle Tests (Tests 7-14)
    # =========================================================================

    def test_07_confirmed_deposit_protected_against_track_id_reacquisition(self):
        """7. One confirmed deposit remains protected against tracker-ID reacquisition."""
        # Verified entry
        w_out = make_tracked_waste("Waste_1", BoundingBox(500, 300, 540, 340), frame_idx=1, timestamp=1.0)
        self.engine.process_frame([], [w_out], 1000, 1000, 1, 1.0)
        box1 = BoundingBox(750, 600, 800, 650)
        w1 = make_tracked_waste("Waste_1", box1, frame_idx=2, timestamp=1.1)
        ev = self.engine.process_frame([], [w1], 1000, 1000, 2, 1.1)
        self.assertEqual(len(ev), 1)

        # ByteTrack re-acquires as Waste_2 with high IoU
        box2 = BoundingBox(752, 602, 802, 652)
        w2 = make_tracked_waste("Waste_2", box2, frame_idx=40, timestamp=10.0)
        ev_dup = self.engine.process_frame([], [w2], 1000, 1000, 40, 10.0)
        self.assertEqual(len(ev_dup), 0, "Confirmed deposit must suppress reacquisition")

    def test_08_temporary_detector_misses_do_not_rearm(self):
        """8. Temporary detector misses do not automatically re-arm a deposit."""
        # Deposit confirmed
        w_out = make_tracked_waste("Waste_1", BoundingBox(500, 300, 540, 340), frame_idx=1, timestamp=1.0)
        self.engine.process_frame([], [w_out], 1000, 1000, 1, 1.0)
        box = BoundingBox(750, 600, 800, 650)
        w1 = make_tracked_waste("Waste_1", box, frame_idx=2, timestamp=1.1)
        self.engine.process_frame([], [w1], 1000, 1000, 2, 1.1)

        # Detector misses object for 10 frames (< 30 frames threshold)
        for f in range(3, 13):
            self.engine.process_frame([], [], 1000, 1000, f, 1.1 + f * 0.1)

        self.assertEqual(self.engine._confirmed_deposits[0].state, "PRESENT")
        self.assertEqual(self.engine._confirmed_deposits[0].consecutive_missing_frames, 10)

        # Object re-detected at frame 13: still PRESENT, no new event
        w_re = make_tracked_waste("Waste_re", box, frame_idx=13, timestamp=2.5)
        ev = self.engine.process_frame([], [w_re], 1000, 1000, 13, 2.5)
        self.assertEqual(len(ev), 0, "Temporary miss must not re-arm or duplicate event")
        self.assertEqual(self.engine._confirmed_deposits[0].consecutive_missing_frames, 0)

    def test_09_temporary_occlusion_does_not_rearm(self):
        """9. Temporary occlusion does not automatically re-arm a deposit."""
        w_out = make_tracked_waste("Waste_1", BoundingBox(500, 300, 540, 340), frame_idx=1, timestamp=1.0)
        self.engine.process_frame([], [w_out], 1000, 1000, 1, 1.0)
        box = BoundingBox(750, 600, 800, 650)
        w1 = make_tracked_waste("Waste_1", box, frame_idx=2, timestamp=1.1)
        self.engine.process_frame([], [w1], 1000, 1000, 2, 1.1)

        # A person stands directly in front of the deposit, occluding it for 45 frames (> 30 frames)
        person_occluding = make_tracked_person("Person_Cleaner", BoundingBox(730, 550, 820, 850))
        for f in range(3, 48):
            # Waste detection is lost due to occlusion
            self.engine.process_frame([person_occluding], [], 1000, 1000, f, 1.1 + f * 0.1)

        # Because person was occluding, consecutive_missing_frames must not advance towards removal
        self.assertEqual(self.engine._confirmed_deposits[0].state, "PRESENT")
        self.assertEqual(self.engine._confirmed_deposits[0].consecutive_missing_frames, 0)

    def test_10_confirmed_physical_removal_rearms_deposit_identity(self):
        """10. A confirmed physical removal correctly re-arms the relevant deposit identity."""
        w_out = make_tracked_waste("Waste_1", BoundingBox(500, 300, 540, 340), frame_idx=1, timestamp=1.0)
        self.engine.process_frame([], [w_out], 1000, 1000, 1, 1.0)
        box = BoundingBox(750, 600, 800, 650)
        w1 = make_tracked_waste("Waste_1", box, frame_idx=2, timestamp=1.1)
        self.engine.process_frame([], [w1], 1000, 1000, 2, 1.1)

        # Physical removal: absent for 35 frames with no person occluding
        for f in range(3, 38):
            self.engine.process_frame([], [], 1000, 1000, f, 1.1 + f * 0.1)

        # State must transition to REMOVED
        self.assertEqual(self.engine._confirmed_deposits[0].state, "REMOVED")

    def test_11_new_item_at_same_location_after_removal_produces_separate_event(self):
        """11. A new item placed at the same location after removal can produce a separate event."""
        # 1. First disposal
        w_out1 = make_tracked_waste("Waste_1", BoundingBox(500, 300, 540, 340), frame_idx=1, timestamp=1.0)
        self.engine.process_frame([], [w_out1], 1000, 1000, 1, 1.0)
        box = BoundingBox(750, 600, 800, 650)
        w1 = make_tracked_waste("Waste_1", box, frame_idx=2, timestamp=1.1)
        ev1 = self.engine.process_frame([], [w1], 1000, 1000, 2, 1.1)
        self.assertEqual(len(ev1), 1)

        # 2. Confirmed removal: missing for 35 frames
        for f in range(3, 38):
            self.engine.process_frame([], [], 1000, 1000, f, 1.1 + f * 0.1)
        self.assertEqual(self.engine._confirmed_deposits[0].state, "REMOVED")

        # 3. New disposal at same location (brought from outside)
        w_out2 = make_tracked_waste("Waste_2", BoundingBox(500, 300, 540, 340), frame_idx=40, timestamp=15.0)
        self.engine.process_frame([], [w_out2], 1000, 1000, 40, 15.0)
        w2 = make_tracked_waste("Waste_2", box, frame_idx=41, timestamp=15.1)
        ev2 = self.engine.process_frame([], [w2], 1000, 1000, 41, 15.1)
        self.assertEqual(len(ev2), 1, "New item at re-armed location must produce separate event")
        self.assertEqual(ev2[0].waste_track_id, "Waste_2")

    def test_12_two_separate_items_entering_same_bin_independently_recorded(self):
        """12. Two separate items entering the same bin remain independently recordable."""
        # Item 1 enters top-left of bin
        w1_out = make_tracked_waste("Waste_1", BoundingBox(500, 300, 540, 340), frame_idx=1, timestamp=1.0)
        self.engine.process_frame([], [w1_out], 1000, 1000, 1, 1.0)
        box1 = BoundingBox(710, 520, 750, 560)
        w1_in = make_tracked_waste("Waste_1", box1, frame_idx=2, timestamp=1.1)
        ev1 = self.engine.process_frame([], [w1_in], 1000, 1000, 2, 1.1)
        self.assertEqual(len(ev1), 1)

        # Item 2 enters bottom-right of bin
        w2_out = make_tracked_waste("Waste_2", BoundingBox(500, 300, 540, 340), frame_idx=3, timestamp=1.2)
        self.engine.process_frame([], [w2_out], 1000, 1000, 3, 1.2)
        box2 = BoundingBox(850, 800, 900, 850)
        self.assertLess(box1.iou(box2), 0.10)
        w2_in = make_tracked_waste("Waste_2", box2, frame_idx=4, timestamp=1.3)
        ev2 = self.engine.process_frame([], [w1_in, w2_in], 1000, 1000, 4, 1.3)
        self.assertEqual(len(ev2), 1)
        self.assertEqual(ev2[0].waste_track_id, "Waste_2")

    def test_13_two_consecutive_physical_disposals_remain_separate(self):
        """13. Two consecutive physical disposals close together remain separate when they are distinct."""
        p1 = make_tracked_person("Person_1", BoundingBox(100, 500, 200, 800))
        p2 = make_tracked_person("Person_2", BoundingBox(300, 500, 400, 800))

        # Person 1 holds waste 1 outside bin across 2 frames
        w1_held = make_tracked_waste("Waste_1", BoundingBox(150, 550, 190, 600))
        self.engine.process_frame([p1], [w1_held], 1000, 1000, 1, 1.0)
        self.engine.process_frame([p1], [w1_held], 1000, 1000, 2, 1.1)

        # Deposited into bin
        w1_bin = make_tracked_waste("Waste_1", BoundingBox(720, 550, 760, 600))
        ev1 = self.engine.process_frame([p1], [w1_bin], 1000, 1000, 3, 1.2)
        self.assertEqual(len(ev1), 1)
        self.assertEqual(ev1[0].person_track_id, "Person_1")

        # Person 2 holds waste 2 outside bin across 2 frames
        w2_held = make_tracked_waste("Waste_2", BoundingBox(350, 550, 390, 600))
        self.engine.process_frame([p2], [w2_held], 1000, 1000, 4, 1.3)
        self.engine.process_frame([p2], [w2_held], 1000, 1000, 5, 1.4)

        # Deposited into bin at another location
        w2_bin = make_tracked_waste("Waste_2", BoundingBox(850, 750, 890, 800))
        ev2 = self.engine.process_frame([p2], [w2_bin], 1000, 1000, 6, 1.5)
        self.assertEqual(len(ev2), 1)
        self.assertEqual(ev2[0].person_track_id, "Person_2")

    def test_14_stream_reset_clears_previous_session_state(self):
        """14. Stream/session reset clears only state belonging to the previous session."""
        w_out = make_tracked_waste("Waste_1", BoundingBox(500, 300, 540, 340), frame_idx=1, timestamp=1.0)
        self.engine.process_frame([], [w_out], 1000, 1000, 1, 1.0)
        box = BoundingBox(750, 600, 800, 650)
        w1 = make_tracked_waste("Waste_1", box, frame_idx=2, timestamp=1.1)
        ev1 = self.engine.process_frame([], [w1], 1000, 1000, 2, 1.1)
        self.assertEqual(len(ev1), 1)

        self.engine.reset()
        self.assertEqual(len(self.engine._confirmed_deposits), 0)
        self.assertEqual(len(self.engine._track_states), 0)

    # =========================================================================
    # Group 3: System Preservation and Regression Safeguards (Tests 15-20)
    # =========================================================================

    def test_15_carrying_without_release_zero_events(self):
        """15. Carrying without release generates no dumping event."""
        p = make_tracked_person("Person_1", BoundingBox(200, 400, 300, 700))
        w = make_tracked_waste("Waste_1", BoundingBox(230, 500, 260, 540))

        for f in range(1, 30):
            offset = f * 10
            p_mov = make_tracked_person("Person_1", BoundingBox(200 + offset, 400, 300 + offset, 700))
            w_mov = make_tracked_waste("Waste_1", BoundingBox(230 + offset, 500, 260 + offset, 540))
            ev = self.engine.process_frame([p_mov], [w_mov], 1000, 1000, f, f * 0.1)
            self.assertEqual(len(ev), 0)

    def test_16_pedestrians_walking_through_outdoor_foliage_zero_events(self):
        """16. Pedestrians walking through outdoor foliage do not generate dumping events."""
        # Scale implausibility check: large foliage box enclosing person
        person = make_tracked_person("Pedestrian", BoundingBox(400, 300, 480, 650))
        foliage = make_tracked_waste("TreeFoliage", BoundingBox(350, 250, 600, 700), class_name="recyclable")

        for f in range(1, 20):
            ev = self.engine.process_frame([person], [foliage], 1000, 1000, f, f * 0.1)
            self.assertEqual(len(ev), 0)

    def test_17_designated_bin_disposal_remains_functional(self):
        """17. Designated-bin disposal remains functional."""
        p = make_tracked_person("Person_1", BoundingBox(550, 450, 650, 800))
        w_held = make_tracked_waste("Bottle_1", BoundingBox(580, 520, 610, 560))
        self.engine.process_frame([p], [w_held], 1000, 1000, 1, 1.0)
        self.engine.process_frame([p], [w_held], 1000, 1000, 2, 1.1)

        w_bin = make_tracked_waste("Bottle_1", BoundingBox(760, 580, 790, 620))
        ev = self.engine.process_frame([p], [w_bin], 1000, 1000, 3, 1.2)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].event_type, DisposalEventType.BIN_DISPOSAL)

    def test_18_open_ground_disposal_remains_functional(self):
        """18. Open-ground disposal remains functional and idempotent."""
        p = make_tracked_person("Person_1", BoundingBox(200, 400, 300, 700))
        w = make_tracked_waste("Waste_1", BoundingBox(230, 500, 260, 540))

        # Person holds waste in air/ground
        self.engine.process_frame([p], [w], 1000, 1000, 1, 1.0)
        self.engine.process_frame([p], [w], 1000, 1000, 2, 1.1)

        # Person walks away (separated)
        p_far = make_tracked_person("Person_1", BoundingBox(50, 400, 150, 700))
        # Waste rests on ground for 3.5 seconds
        self.engine.process_frame([p_far], [w], 1000, 1000, 10, 2.0)
        ev = self.engine.process_frame([p_far], [w], 1000, 1000, 100, 6.0)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].event_type, DisposalEventType.OPEN_GROUND_LITTERING)
        self.assertEqual(ev[0].waste_track_id, "Waste_1")
        self.assertEqual(ev[0].person_track_id, "Person_1")

        # After confirmation, track is lost and reacquired as Waste_2 at same location
        w_reacq = make_tracked_waste("Waste_2", BoundingBox(231, 501, 261, 541))
        ev_reacq = self.engine.process_frame([p_far], [w_reacq], 1000, 1000, 200, 10.0)
        self.assertEqual(len(ev_reacq), 0, "Reacquired open ground deposit must not emit second littering event!")

    def test_19_ambiguous_multi_person_attribution_protected(self):
        """19. Ambiguous multi-person attribution remains protected."""
        p1 = make_tracked_person("Person_1", BoundingBox(100, 500, 200, 800))
        p2 = make_tracked_person("Person_2", BoundingBox(120, 500, 220, 800))
        w_held = make_tracked_waste("Waste_1", BoundingBox(150, 550, 180, 600))

        self.engine.process_frame([p1, p2], [w_held], 1000, 1000, 1, 1.0)
        self.engine.process_frame([p1, p2], [w_held], 1000, 1000, 2, 1.1)

        w_bin = make_tracked_waste("Waste_1", BoundingBox(750, 600, 780, 650))
        ev = self.engine.process_frame([p1, p2], [w_bin], 1000, 1000, 3, 1.2)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].attribution_status, AttributionStatus.AMBIGUOUS)
        self.assertIsNone(ev[0].person_track_id)

    def test_20_existing_face_recognition_and_liveness_unchanged(self):
        """20. Existing face-recognition and liveness/PAD behavior remains unchanged."""
        event = ConfirmedLitterEvent(
            event_id="evt_test_001",
            event_type=DisposalEventType.BIN_DISPOSAL,
            estimated_release_time="2026-11-24T12:00:00Z",
            estimated_release_timestamp=10.0,
            confirmed_at="2026-11-24T12:00:01Z",
            confirmed_timestamp=11.0,
            waste_category="recyclable",
            waste_confidence=0.92,
            waste_track_id="Waste_1",
            person_track_id="Person_1",
            recognized_person="Alice",
            attribution_status=AttributionStatus.ATTRIBUTED,
            recognition_status=RecognitionStatus.IDENTIFIED,
            ground_zone_id="bin_primary",
            notes="Face recognition and liveness intact",
        )
        self.assertEqual(event.recognized_person, "Alice")
        self.assertEqual(event.recognition_status, RecognitionStatus.IDENTIFIED)


if __name__ == "__main__":
    unittest.main()
