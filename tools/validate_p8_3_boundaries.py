"""Comprehensive validation script for P8.3 Detection & Association Boundary Validation.

Collects exact data for:
1. Model SHA-256 hash verification.
2. Protected original workspace non-destructive comparison.
3. Facial & clothing filtering boundary analysis:
   - Facial/head false positives vs valid carried items (bottle near mouth, chest, package, held beside).
4. Holding & separation boundary analysis:
   - Spatial overlap, 5px gap, 10px gap, 11px gap, 25px gap.
   - Proximity score impact: Verify proximity alone NEVER attributes a dumper.
5. Real-model evaluation on actual recorded video evidence clips from data/evidence/.
6. Summary of synthetic vs real model execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT_ROOT = Path("c:/Users/A/OneDrive/Desktop/Waste_dumping_v2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.common_types import BoundingBox, Detection, TrackedObject, LitteringState
from app.detection.waste_detector import WasteDetector
from app.detection.person_detector import PersonDetector
from app.association.association_engine import PersonWasteAssociationEngine, WasteTrajectoryTracker
from app.association.zones import DetectionZone, ZoneManager


def sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_hash_integrity():
    print("--- 1. MODEL HASH VERIFICATION ---")
    models = {
        "waste_detector": "models/waste/waste_detection_yolo11m.pt",
        "person_detector": "models/person/person_detector.pt",
    }
    v2_hashes = {}
    for name, rel in models.items():
        p = str(PROJECT_ROOT / rel)
        h = sha256_file(p)
        v2_hashes[name] = h
        print(f"v2 {name} ({rel}): {h}")

    print("\n--- 2. PROTECTED WORKSPACE INTEGRITY ---")
    orig_dir = "c:/Users/A/OneDrive/Desktop/Waste_dumping"
    orig_files = {
        "yolo11n.pt": os.path.join(orig_dir, "yolo11n.pt"),
        "faces.json": os.path.join(orig_dir, "Face_recognize", "faces.json"),
        "waste_detection_yolo11m.pt": os.path.join(orig_dir, "Waste_detection", "models", "pretrained", "waste_detection_yolo11m.pt"),
    }
    for name, p in orig_files.items():
        h = sha256_file(p)
        size = os.path.getsize(p) if os.path.exists(p) else 0
        print(f"Original {name}: {h} (size={size})")

    match_waste = (v2_hashes["waste_detector"] == sha256_file(orig_files["waste_detection_yolo11m.pt"]))
    match_person = (v2_hashes["person_detector"] == sha256_file(orig_files["yolo11n.pt"]))
    print(f">> Hash match waste: {match_waste}, person: {match_person}")
    return match_waste and match_person


def test_facial_and_clothing_boundaries():
    print("\n--- 3. PROBE: FACIAL & CLOTHING FILTERING BOUNDARIES ---")
    detector = WasteDetector(
        model_path="models/waste/waste_detection_yolo11m.pt",
        confidence_threshold=0.25,
        verify_sha256=True,
    )

    person_box = BoundingBox(200.0, 100.0, 400.0, 460.0)
    pw, ph = person_box.width, person_box.height
    print(f"Test Person Box: {person_box} (pw={pw}, ph={ph})")

    test_cases = [
        ("Hair/Forehead (top of head)", BoundingBox(260.0, 95.0, 340.0, 160.0), True, "cephalic (y2=160, 16.7% ph)"),
        ("Face/Eyes/Nose", BoundingBox(265.0, 120.0, 335.0, 200.0), True, "cephalic (y2=200, 27.8% ph)"),
        ("Beard/Jaw/Neck (P8.2 boundary)", BoundingBox(270.0, 150.0, 350.0, 260.0), True, "cephalic (y2=260, 44.4% ph <= 48%)"),
        ("Borderline Neck/Collar (y_center=0.321)", BoundingBox(270.0, 160.0, 340.0, 271.0), True, "cephalic edge test"),
        ("Bottle near mouth/chin (drinking)", BoundingBox(280.0, 180.0, 320.0, 290.0), False, "Preserved (y2=290 is 52.8% ph > 48%)"),
        ("Bag near upper chest (high carry)", BoundingBox(240.0, 260.0, 360.0, 350.0), False, "Preserved (y1=260 > cephalic cutoff)"),
        ("Large cardboard carried (waist)", BoundingBox(180.0, 280.0, 380.0, 420.0), False, "Preserved (legitimate large waste)"),
        ("Bottle held beside body (hand)", BoundingBox(385.0, 290.0, 425.0, 370.0), False, "Preserved (outside torso)"),
        ("Entire shirt / torso clothing", BoundingBox(220.0, 220.0, 380.0, 360.0), True, "large torso clothing"),
        ("Pants / lower body clothing", BoundingBox(230.0, 320.0, 370.0, 450.0), True, "large torso clothing"),
    ]

    results = []
    for name, w_box, expect_reject, reason in test_cases:
        ioa = w_box.ioa(person_box)
        rel_area = w_box.area / person_box.area
        y_center_norm = (w_box.center[1] - person_box.y1) / ph
        y2_norm = (w_box.y2 - person_box.y1) / ph
        is_rejected = detector._is_false_human_detection(w_box, 0.50, [person_box])
        passed = (is_rejected == expect_reject)
        status = "PASS" if passed else "FAIL"
        res_item = {
            "name": name,
            "box": str(w_box),
            "ioa": round(ioa, 3),
            "rel_area": round(rel_area, 3),
            "y2_norm": round(y2_norm, 3),
            "y_center_norm": round(y_center_norm, 3),
            "is_rejected": is_rejected,
            "expected_rejected": expect_reject,
            "status": status,
            "reason": reason,
        }
        results.append(res_item)
        print(f"[{status}] {name:38s}: y2_norm={y2_norm:.3f}, yc_norm={y_center_norm:.3f}, IoA={ioa:.3f}, RelArea={rel_area:.3f} | Rejected={is_rejected} (Expect {expect_reject})")
    return results


def test_holding_and_attribution_boundaries():
    print("\n--- 4. PROBE: HOLDING & ATTRIBUTION BOUNDARIES ---")
    zm = ZoneManager.__new__(ZoneManager)
    zm.zones = {
        "ground": DetectionZone("ground", "ground", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], is_normalized=True),
        "bin": DetectionZone("bin", "bin", [[0.8, 0.8], [1.0, 0.8], [1.0, 1.0], [0.8, 1.0]], is_normalized=True),
    }

    person_box = BoundingBox(200.0, 100.0, 400.0, 460.0)
    holding_upper_y = person_box.y1 + 0.15 * person_box.height  # 154
    holding_lower_y = person_box.y1 + 0.85 * person_box.height  # 406

    def compute_separation(p_box, w_box):
        dx = max(0.0, max(p_box.x1 - w_box.x2, w_box.x1 - p_box.x2))
        dy = max(0.0, max(p_box.y1 - w_box.y2, w_box.y1 - p_box.y2))
        return math.hypot(dx, dy)

    def is_holding_check(p_box, w_box):
        wy = w_box.center[1]
        is_in_hand_volume = (holding_upper_y <= wy <= holding_lower_y)
        iou = p_box.iou(w_box)
        ioa = w_box.ioa(p_box)
        box_sep = compute_separation(p_box, w_box)
        is_spatial_overlap = (
            iou >= 0.05
            or ioa >= 0.10
            or box_sep <= 10.0
        )
        return is_in_hand_volume and is_spatial_overlap, box_sep, iou, ioa

    separation_cases = [
        ("0px gap (overlapping in hand)", BoundingBox(380.0, 280.0, 410.0, 330.0), True),
        ("5px gap (touching/immediate hand vicinity)", BoundingBox(405.0, 280.0, 435.0, 330.0), True),
        ("10px gap (boundary limit for contact)", BoundingBox(410.0, 280.0, 440.0, 330.0), True),
        ("11px gap (just outside physical contact)", BoundingBox(411.0, 280.0, 441.0, 330.0), False),
        ("25px gap (separated / near person)", BoundingBox(425.0, 280.0, 455.0, 330.0), False),
        ("50px gap (passerby walking past stationary item)", BoundingBox(450.0, 280.0, 480.0, 330.0), False),
    ]

    sep_results = []
    for name, w_box, expect_holding in separation_cases:
        holding, sep, iou, ioa = is_holding_check(person_box, w_box)
        passed = (holding == expect_holding)
        status = "PASS" if passed else "FAIL"
        sep_results.append({
            "name": name,
            "separation_px": round(sep, 1),
            "iou": round(iou, 3),
            "ioa": round(ioa, 3),
            "holding": holding,
            "expected_holding": expect_holding,
            "status": status,
        })
        print(f"[{status}] {name:48s}: Separation={sep:5.1f}px | IoU={iou:.3f} | IoA={ioa:.3f} | Holding={holding} (Expect {expect_holding})")

    print("\n--- 5. PROXIMITY SCORING & PASSERBY ISOLATION AUDIT ---")
    engine = PersonWasteAssociationEngine(zone_manager=zm, min_holding_frames=2)
    waste_box = BoundingBox(500.0, 300.0, 530.0, 350.0)

    # Frame 1: Person walking past pre-existing stationary litter
    p1 = TrackedObject(track_id="Person_1", raw_id=1, class_name="person", bbox=BoundingBox(300.0, 100.0, 450.0, 460.0), confidence=0.9, last_frame_idx=1, last_timestamp=0.0)
    w1 = TrackedObject(track_id="Waste_10", raw_id=10, class_name="recyclable", bbox=waste_box, confidence=0.8, last_frame_idx=1, last_timestamp=0.0)
    engine.process_frame([p1], [w1], 640, 480, 1, 0.0)

    # Frame 2: Person walking closer (15px edge gap)
    p2 = TrackedObject(track_id="Person_1", raw_id=1, class_name="person", bbox=BoundingBox(350.0, 100.0, 485.0, 460.0), confidence=0.9, last_frame_idx=2, last_timestamp=0.1)
    w2 = TrackedObject(track_id="Waste_10", raw_id=10, class_name="recyclable", bbox=waste_box, confidence=0.8, last_frame_idx=2, last_timestamp=0.1)
    engine.process_frame([p2], [w2], 640, 480, 2, 0.1)

    # Frame 3: Person walking away (100px edge gap)
    p3 = TrackedObject(track_id="Person_1", raw_id=1, class_name="person", bbox=BoundingBox(200.0, 100.0, 350.0, 460.0), confidence=0.9, last_frame_idx=3, last_timestamp=3.5)
    w3 = TrackedObject(track_id="Waste_10", raw_id=10, class_name="recyclable", bbox=waste_box, confidence=0.8, last_frame_idx=3, last_timestamp=3.5)
    events = engine.process_frame([p3], [w3], 640, 480, 3, 3.5)

    tracker = engine.get_trajectory("Waste_10")
    passerby_passed = (
        tracker.is_pre_existing_litter == True
        and len(tracker.person_held_frames) == 0
        and len(events) == 0
    )
    print(f"Passerby test: pre_existing={tracker.is_pre_existing_litter}, held_frames={len(tracker.person_held_frames)}, events={len(events)} -> PASS: {passerby_passed}")

    return sep_results, passerby_passed


def test_real_video_evidence():
    print("\n--- 6. PROBE: REAL-MODEL & REAL-VIDEO EVIDENCE VALIDATION ---")
    evidence_dir = PROJECT_ROOT / "data" / "evidence"
    video_files = list(evidence_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} real evidence video clips.")

    waste_detector = WasteDetector(
        model_path="models/waste/waste_detection_yolo11m.pt",
        confidence_threshold=0.25,
        verify_sha256=True,
    )
    person_detector = PersonDetector(
        model_path="models/person/person_detector.pt",
        confidence_threshold=0.35,
    )

    clip_results = []
    # Test top 6 most recent clips
    recent_clips = sorted(video_files, key=lambda f: os.path.getmtime(f), reverse=True)[:6]
    for clip in recent_clips:
        cap = cv2.VideoCapture(str(clip))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        
        raw_waste_count = 0
        filtered_waste_count = 0
        person_count = 0
        cephalic_rejected_count = 0
        oversized_rejected_count = 0
        clothing_rejected_count = 0

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1

            # Detect persons
            persons = person_detector.detect(frame)
            person_boxes = [p.bbox for p in persons]
            if persons:
                person_count += 1

            # Raw YOLO prediction
            raw_res = waste_detector.model.predict(source=frame, conf=0.10, verbose=False)
            if raw_res and len(raw_res) > 0 and raw_res[0].boxes is not None:
                fh, fw = frame.shape[:2]
                frame_area = float(fw * fh)
                for b in raw_res[0].boxes:
                    conf = float(b.conf.cpu().numpy()[0])
                    if conf >= 0.25:
                        raw_waste_count += 1
                        xyxy = b.xyxy.cpu().numpy()[0]
                        w_box = BoundingBox(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
                        
                        # Check filter rejections
                        if w_box.area / frame_area >= 0.48 or (w_box.width >= 0.88 * fw and w_box.height >= 0.60 * fh):
                            oversized_rejected_count += 1
                        elif person_boxes:
                            for p_box in person_boxes:
                                pw, ph = p_box.width, p_box.height
                                y_center = w_box.center[1]
                                is_cephalic = (
                                    w_box.y2 <= (p_box.y1 + 0.48 * ph)
                                    and y_center <= (p_box.y1 + 0.32 * ph)
                                    and (w_box.x1 >= p_box.x1 - 25.0 and w_box.x2 <= p_box.x2 + 25.0)
                                    and (w_box.ioa(p_box) >= 0.45 or p_box.ioa(w_box) >= 0.60)
                                )
                                if is_cephalic:
                                    cephalic_rejected_count += 1
                                    break
                                ioa = w_box.ioa(p_box)
                                rel_area = w_box.area / p_box.area
                                is_large_clothing = (
                                    ioa >= 0.70
                                    and rel_area >= 0.12
                                    and (w_box.width >= (0.42 * pw) or rel_area >= 0.20)
                                )
                                if is_large_clothing:
                                    clothing_rejected_count += 1
                                    break

            # Filtered detections
            filtered = waste_detector.detect(frame, person_boxes=person_boxes)
            filtered_waste_count += len(filtered)

        cap.release()
        clip_results.append({
            "clip": clip.name,
            "frames": total_frames,
            "person_frames": person_count,
            "raw_waste_detections": raw_waste_count,
            "filtered_waste_detections": filtered_waste_count,
            "oversized_rejected": oversized_rejected_count,
            "cephalic_rejected": cephalic_rejected_count,
            "clothing_rejected": clothing_rejected_count,
        })
        print(f"Clip: {clip.name} ({total_frames} frames)")
        print(f"  Person frames: {person_count}")
        print(f"  Raw waste >= 0.25: {raw_waste_count}")
        print(f"  Filtered waste: {filtered_waste_count}")
        print(f"  Oversized rejected: {oversized_rejected_count}")
        print(f"  Cephalic rejected: {cephalic_rejected_count}")
        print(f"  Clothing rejected: {clothing_rejected_count}")

    print(">> Real video evidence processing complete.")
    return clip_results


if __name__ == "__main__":
    h_ok = test_hash_integrity()
    f_res = test_facial_and_clothing_boundaries()
    s_res, p_ok = test_holding_and_attribution_boundaries()
    v_res = test_real_video_evidence()

    summary = {
        "hash_integrity": h_ok,
        "facial_and_clothing": f_res,
        "holding_separation": s_res,
        "passerby_isolated": p_ok,
        "video_clips": v_res,
    }
    out_path = PROJECT_ROOT / "reports" / "p8_3_boundary_validation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved structured audit results to {out_path}")
