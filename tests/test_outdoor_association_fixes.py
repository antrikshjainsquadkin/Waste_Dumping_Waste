"""Regression tests for outdoor false-positive elimination and association fixes.

Validates:
1. Person walking past large trees / background shadows does not trigger holding or littering.
2. Person walking past pre-existing litter does not trigger holding or false event.
3. Person standing near existing waste does not trigger false event.
4. Person carrying waste continuously without release produces no event.
5. Genuine handling, release, and stationary delay generates single OPEN_GROUND_LITTERING.
6. Genuine waste deposited into bin generates BIN_DISPOSAL, while overhead canopy is rejected.
7. Multi-person proximity during release preserves AMBIGUOUS attribution.
8. Large legitimate waste items (cartons, large bags) are not improperly filtered.
9. Single disposal action does not produce duplicate events across track transitions.
"""

import unittest
from app.common_types import AttributionStatus, BoundingBox, DisposalEventType, TrackedObject
from app.association.zones import DetectionZone, ZoneManager
from app.association.association_engine import PersonWasteAssociationEngine


def make_tracked_obj(
    track_id: str,
    bbox: BoundingBox,
    class_name: str = "recyclable",
    confidence: float = 0.8,
    frame_idx: int = 0,
    timestamp: float = 0.0,
) -> TrackedObject:
    """Helper to instantiate TrackedObject with all required fields."""
    return TrackedObject(
        track_id=track_id,
        raw_id=1,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        last_frame_idx=frame_idx,
        last_timestamp=timestamp,
    )


def make_dummy_zone_manager() -> ZoneManager:
    """Create in-memory ZoneManager with bin and ground zones."""
    zm = ZoneManager.__new__(ZoneManager)
    zm.zones = {
        "bin_primary": DetectionZone(
            zone_id="bin_primary",
            zone_type="bin",
            polygon=[[0.70, 0.50], [0.85, 0.50], [0.85, 0.85], [0.70, 0.85]],
            is_normalized=True,
            description="Outdoor Bin",
        ),
        "open_ground": DetectionZone(
            zone_id="open_ground",
            zone_type="ground",
            polygon=[[0.0, 0.30], [1.0, 0.30], [1.0, 1.0], [0.0, 1.0]],
            is_normalized=True,
            description="Ground Plane",
        ),
    }
    return zm


class TestOutdoorAssociationFixes(unittest.TestCase):
    """Regression test suite for outdoor detection and association fixes."""

    def setUp(self):
        self.zm = make_dummy_zone_manager()
        self.engine = PersonWasteAssociationEngine(
            zone_manager=self.zm,
            holding_max_distance_px=160.0,
            holding_iou_threshold=0.05,
            min_holding_frames=2,
            separation_distance_delta_px=20.0,
            stationary_duration_seconds=3.0,
            event_cooldown_seconds=4.0,
            ambiguity_margin_px=35.0,
        )
        self.frame_w = 1920
        self.frame_h = 1080

    def test_01_person_walks_past_shadow_canopy_no_event(self):
        """Outdoor ambient canopy/shadow (Waste_4 in P10_EXT_04) enclosing pedestrian must not be held."""
        # Tree canopy / shadow spans wide path (width 768, height 146)
        canopy_bbox = BoundingBox(x1=316.0, y1=527.0, x2=1084.0, y2=673.0)
        waste = make_tracked_obj("Waste_Canopy", canopy_bbox, class_name="recyclable", confidence=0.6)

        # Pedestrian walks along path through the canopy
        events = []
        for i in range(10):
            t = i * 0.1
            px = 500.0 + i * 20.0
            person_bbox = BoundingBox(x1=px, y1=540.0, x2=px + 40.0, y2=660.0)
            person = make_tracked_obj("Person_Walker", person_bbox, class_name="person", confidence=0.85, frame_idx=i, timestamp=t)
            waste.last_frame_idx = i
            waste.last_timestamp = t
            evs = self.engine.process_frame([person], [waste], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        # Person exits canopy, walks away for 5 seconds (50 frames)
        for i in range(10, 60):
            t = i * 0.1
            px = 700.0 + (i - 10) * 15.0
            person_bbox = BoundingBox(x1=px, y1=540.0, x2=px + 40.0, y2=660.0)
            person = make_tracked_obj("Person_Walker", person_bbox, class_name="person", confidence=0.85, frame_idx=i, timestamp=t)
            waste.last_frame_idx = i
            waste.last_timestamp = t
            evs = self.engine.process_frame([person], [waste], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        # Verify: No false holding was registered, so no dumping event was generated
        self.assertEqual(len(events), 0, "Walking past ambient canopy/shadow must NOT generate any littering event")

    def test_02_person_walks_past_pre_existing_litter_no_event(self):
        """Walking past pre-existing stationary litter does not trigger holding or event."""
        litter_bbox = BoundingBox(x1=500.0, y1=700.0, x2=530.0, y2=730.0)
        litter = make_tracked_obj("Waste_Ground", litter_bbox, class_name="recyclable", confidence=0.7)

        events = []
        # First 3 frames: litter alone, establishing pre-existing status
        for i in range(3):
            t = i * 0.1
            litter.last_frame_idx = i
            litter.last_timestamp = t
            evs = self.engine.process_frame([], [litter], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        # Person approaches and walks right next to it
        for i in range(3, 15):
            t = i * 0.1
            px = 450.0 + (i - 3) * 10.0
            p_bbox = BoundingBox(x1=px, y1=620.0, x2=px + 45.0, y2=750.0)
            person = make_tracked_obj("Person_1", p_bbox, class_name="person", confidence=0.85, frame_idx=i, timestamp=t)
            litter.last_frame_idx = i
            litter.last_timestamp = t
            evs = self.engine.process_frame([person], [litter], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        # Person moves far away
        for i in range(15, 60):
            t = i * 0.1
            px = 600.0 + (i - 15) * 10.0
            p_bbox = BoundingBox(x1=px, y1=620.0, x2=px + 45.0, y2=750.0)
            person = make_tracked_obj("Person_1", p_bbox, class_name="person", confidence=0.85, frame_idx=i, timestamp=t)
            litter.last_frame_idx = i
            litter.last_timestamp = t
            evs = self.engine.process_frame([person], [litter], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        self.assertEqual(len(events), 0, "Walking past pre-existing ground litter must not emit events")

    def test_03_person_stands_near_existing_waste_no_event(self):
        """Standing stationary near existing waste does not claim holding or emit events."""
        litter_bbox = BoundingBox(x1=600.0, y1=800.0, x2=630.0, y2=830.0)
        litter = make_tracked_obj("Waste_Stationary", litter_bbox, class_name="plastic", confidence=0.75)

        # Unheld initialization
        for i in range(3):
            t = i * 0.1
            litter.last_frame_idx = i
            litter.last_timestamp = t
            self.engine.process_frame([], [litter], self.frame_w, self.frame_h, i, t)

        events = []
        # Person stands 30px away for 40 frames (4 seconds)
        for i in range(3, 45):
            t = i * 0.1
            p_bbox = BoundingBox(x1=640.0, y1=700.0, x2=690.0, y2=850.0)
            person = make_tracked_obj("Person_Stand", p_bbox, class_name="person", confidence=0.9, frame_idx=i, timestamp=t)
            litter.last_frame_idx = i
            litter.last_timestamp = t
            evs = self.engine.process_frame([person], [litter], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        self.assertEqual(len(events), 0, "Standing near stationary waste must not trigger events")

    def test_04_person_carries_waste_without_release_no_event(self):
        """Person carries waste continuously across scene without releasing: no event emitted."""
        events = []
        for i in range(50):
            t = i * 0.1
            px = 100.0 + i * 15.0
            p_bbox = BoundingBox(x1=px, y1=500.0, x2=px + 60.0, y2=700.0)
            w_bbox = BoundingBox(x1=px + 40.0, y1=580.0, x2=px + 65.0, y2=615.0)
            person = make_tracked_obj("Person_Carry", p_bbox, class_name="person", confidence=0.88, frame_idx=i, timestamp=t)
            waste = make_tracked_obj("Waste_Carried", w_bbox, class_name="plastic", confidence=0.72, frame_idx=i, timestamp=t)

            evs = self.engine.process_frame([person], [waste], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        self.assertEqual(len(events), 0, "Carrying waste without release must produce 0 events")

    def test_05_person_genuinely_handles_and_releases_waste_emits_event(self):
        """Legitimate carry, release, and stationary delay must emit OPEN_GROUND_LITTERING."""
        events = []
        # 1. Carried for 5 frames
        for i in range(5):
            t = i * 0.1
            p_bbox = BoundingBox(x1=300.0, y1=500.0, x2=360.0, y2=700.0)
            w_bbox = BoundingBox(x1=340.0, y1=580.0, x2=365.0, y2=615.0)
            p = make_tracked_obj("Person_1", p_bbox, class_name="person", confidence=0.9, frame_idx=i, timestamp=t)
            w = make_tracked_obj("Waste_Drop", w_bbox, class_name="plastic", confidence=0.8, frame_idx=i, timestamp=t)
            evs = self.engine.process_frame([p], [w], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        # 2. Release: waste drops to ground (y1=750), person walks away
        drop_bbox = BoundingBox(x1=340.0, y1=750.0, x2=365.0, y2=785.0)

        # Person moves away over 4 seconds (40 frames)
        for i in range(5, 50):
            t = i * 0.1
            px = 300.0 + (i - 5) * 15.0
            p_bbox = BoundingBox(x1=px, y1=500.0, x2=px + 60.0, y2=700.0)
            p = make_tracked_obj("Person_1", p_bbox, class_name="person", confidence=0.9, frame_idx=i, timestamp=t)
            w_dropped = make_tracked_obj("Waste_Drop", drop_bbox, class_name="plastic", confidence=0.8, frame_idx=i, timestamp=t)
            evs = self.engine.process_frame([p], [w_dropped], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        self.assertEqual(len(events), 1, "Genuine disposal must confirm exactly 1 event")
        self.assertEqual(events[0].event_type, DisposalEventType.OPEN_GROUND_LITTERING)
        self.assertEqual(events[0].person_track_id, "Person_1")

    def test_06_bin_disposal_functional_and_canopy_rejected(self):
        """Waste inside bin polygon confirms BIN_DISPOSAL; canopy merely touching bin boundary is rejected."""
        # 1. Canopy touching bin: bottom center touches bin, but top touches sky and center is outside
        # bin_primary is normalized [0.70, 0.50] to [0.85, 0.85] -> pixel [1344, 540] to [1632, 918]
        canopy_bbox = BoundingBox(x1=1200.0, y1=50.0, x2=1600.0, y2=600.0)
        canopy = make_tracked_obj("Waste_Canopy_Bin", canopy_bbox, class_name="recyclable", confidence=0.6)
        evs_canopy = self.engine.process_frame([], [canopy], self.frame_w, self.frame_h, 0, 0.0)
        self.assertEqual(len(evs_canopy), 0, "Canopy whose center is outside bin must not trigger BIN_DISPOSAL")

        # 2. Genuine deposit into bin: both bottom_center and center are well inside bin polygon
        person = make_tracked_obj("Person_Bin", BoundingBox(1350.0, 550.0, 1420.0, 850.0), class_name="person", confidence=0.9, frame_idx=1, timestamp=0.1)
        bin_waste_bbox = BoundingBox(x1=1400.0, y1=600.0, x2=1450.0, y2=660.0)
        bin_waste = make_tracked_obj("Waste_In_Bin", bin_waste_bbox, class_name="recyclable", confidence=0.8, frame_idx=1, timestamp=0.1)
        evs_bin = self.engine.process_frame([person], [bin_waste], self.frame_w, self.frame_h, 1, 0.1)
        self.assertEqual(len(evs_bin), 1, "Genuine deposit into bin must confirm BIN_DISPOSAL")
        self.assertEqual(evs_bin[0].event_type, DisposalEventType.BIN_DISPOSAL)

    def test_07_multiple_people_ambiguous_responsibility(self):
        """When two people are equally close to release and neither dominates score, attribution is AMBIGUOUS."""
        events = []
        # Hand-holding position between Person_A and Person_B at torso height
        w_hold = BoundingBox(x1=495.0, y1=670.0, x2=515.0, y2=695.0)

        # Both people hold/touch waste for 3 frames
        for i in range(3):
            t = i * 0.1
            waste = make_tracked_obj("Waste_Ambiguous", w_hold, class_name="plastic", confidence=0.8, frame_idx=i, timestamp=t)
            p1_box = BoundingBox(x1=450.0, y1=600.0, x2=510.0, y2=780.0)
            p2_box = BoundingBox(x1=500.0, y1=600.0, x2=560.0, y2=780.0)
            p1 = make_tracked_obj("Person_A", p1_box, class_name="person", confidence=0.85, frame_idx=i, timestamp=t)
            p2 = make_tracked_obj("Person_B", p2_box, class_name="person", confidence=0.85, frame_idx=i, timestamp=t)
            evs = self.engine.process_frame([p1, p2], [waste], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        # Waste dropped to ground; both walk away in opposite directions, leaving waste stationary
        w_ground = BoundingBox(x1=500.0, y1=800.0, x2=520.0, y2=825.0)
        for i in range(3, 40):
            t = i * 0.1
            waste = make_tracked_obj("Waste_Ambiguous", w_ground, class_name="plastic", confidence=0.8, frame_idx=i, timestamp=t)
            p1_box = BoundingBox(x1=450.0 - (i - 3) * 10, y1=600.0, x2=510.0 - (i - 3) * 10, y2=780.0)
            p2_box = BoundingBox(x1=500.0 + (i - 3) * 10, y1=600.0, x2=560.0 + (i - 3) * 10, y2=780.0)
            p1 = make_tracked_obj("Person_A", p1_box, class_name="person", confidence=0.85, frame_idx=i, timestamp=t)
            p2 = make_tracked_obj("Person_B", p2_box, class_name="person", confidence=0.85, frame_idx=i, timestamp=t)
            evs = self.engine.process_frame([p1, p2], [waste], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        self.assertEqual(len(events), 1, "Must confirm 1 event")
        self.assertEqual(events[0].attribution_status, AttributionStatus.AMBIGUOUS, "Equal proximity must result in AMBIGUOUS attribution")

    def test_08_large_legitimate_waste_objects_not_filtered(self):
        """Large legitimate waste items (e.g., cardboard boxes, large trash bags) must be held and confirmed."""
        events = []
        for i in range(4):
            t = i * 0.1
            p_box = BoundingBox(x1=200.0, y1=400.0, x2=280.0, y2=600.0)
            w_box = BoundingBox(x1=260.0, y1=460.0, x2=350.0, y2=550.0)
            p = make_tracked_obj("Person_Clean", p_box, class_name="person", confidence=0.9, frame_idx=i, timestamp=t)
            w = make_tracked_obj("Waste_LargeBag", w_box, class_name="recyclable", confidence=0.85, frame_idx=i, timestamp=t)
            evs = self.engine.process_frame([p], [w], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        # Bag left on ground, person leaves
        bag_ground = BoundingBox(x1=260.0, y1=700.0, x2=350.0, y2=790.0)
        for i in range(4, 45):
            t = i * 0.1
            px = 200.0 + (i - 4) * 15.0
            p_box = BoundingBox(x1=px, y1=400.0, x2=px + 80.0, y2=600.0)
            p = make_tracked_obj("Person_Clean", p_box, class_name="person", confidence=0.9, frame_idx=i, timestamp=t)
            w_ground = make_tracked_obj("Waste_LargeBag", bag_ground, class_name="recyclable", confidence=0.85, frame_idx=i, timestamp=t)
            evs = self.engine.process_frame([p], [w_ground], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        self.assertEqual(len(events), 1, "Large legitimate waste items must confirm dumping event")
        self.assertEqual(events[0].event_type, DisposalEventType.OPEN_GROUND_LITTERING)
        self.assertEqual(events[0].person_track_id, "Person_Clean")

    def test_09_single_disposal_not_duplicated_by_unrelated_track_transitions(self):
        """Once confirmed, a disposal event does not re-fire when new people walk past."""
        events = []
        # Frame 0 to 4: carry and drop
        for i in range(4):
            t = i * 0.1
            p_box = BoundingBox(x1=100.0, y1=400.0, x2=160.0, y2=600.0)
            w_box = BoundingBox(x1=140.0, y1=480.0, x2=165.0, y2=515.0)
            p = make_tracked_obj("Person_1", p_box, class_name="person", confidence=0.9, frame_idx=i, timestamp=t)
            w = make_tracked_obj("Waste_Item", w_box, class_name="plastic", confidence=0.8, frame_idx=i, timestamp=t)
            evs = self.engine.process_frame([p], [w], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        # Dropped on ground, Person_1 leaves
        ground_box = BoundingBox(x1=140.0, y1=750.0, x2=165.0, y2=785.0)
        for i in range(4, 40):
            t = i * 0.1
            px = 100.0 + (i - 4) * 20.0
            p_box = BoundingBox(x1=px, y1=400.0, x2=px + 60.0, y2=600.0)
            p = make_tracked_obj("Person_1", p_box, class_name="person", confidence=0.9, frame_idx=i, timestamp=t)
            w_ground = make_tracked_obj("Waste_Item", ground_box, class_name="plastic", confidence=0.8, frame_idx=i, timestamp=t)
            evs = self.engine.process_frame([p], [w_ground], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        self.assertEqual(len(events), 1, "Initial event confirmed")

        # Now unrelated Person_2 and Person_3 walk past over the next 50 frames
        for i in range(40, 90):
            t = i * 0.1
            px = 80.0 + (i - 40) * 15.0
            p2 = make_tracked_obj("Person_2", BoundingBox(x1=px, y1=600.0, x2=px + 50.0, y2=780.0), class_name="person", confidence=0.85, frame_idx=i, timestamp=t)
            w_ground = make_tracked_obj("Waste_Item", ground_box, class_name="plastic", confidence=0.8, frame_idx=i, timestamp=t)
            evs = self.engine.process_frame([p2], [w_ground], self.frame_w, self.frame_h, i, t)
            events.extend(evs)

        self.assertEqual(len(events), 1, "Unrelated pedestrian transitions must not generate duplicate events")


if __name__ == "__main__":
    unittest.main()
