"""Probe script to analyze person and waste detections in real evidence clips."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from ultralytics import YOLO

from app.common_types import BoundingBox
from app.detection.person_detector import PersonDetector
from app.detection.waste_detector import WasteDetector

person_detector = PersonDetector("models/person/person_detector.pt")
waste_detector = WasteDetector("models/waste/waste_detection_yolo11m.pt", verify_sha256=False)

clips = [
    "data/evidence/evt_20261119_185229_0001_evidence.mp4",
    "data/evidence/evt_20261119_185408_0005_evidence.mp4",
    "data/evidence/evt_20261120_110246_0001_evidence.mp4",
]

for cp in clips:
    p = Path(cp)
    if not p.is_file():
        print(f"File not found: {cp}")
        continue
    cap = cv2.VideoCapture(str(p))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n==================================================")
    print(f"Analyzing {p.name} ({n_frames} frames)")
    print(f"==================================================")
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f_idx += 1
        p_dets = person_detector.detect(frame)
        p_boxes = [p.bbox for p in p_dets]
        
        # Raw waste before filter
        w_raw = waste_detector.detect(frame, person_boxes=None)
        # Filtered waste after filter
        w_filtered = waste_detector.detect(frame, person_boxes=p_boxes)
        
        # Check details for each raw detection
        details = []
        for wd in w_raw:
            wb = wd.bbox
            suppressed = False
            reasons = []
            for pb in p_boxes:
                iou = wb.iou(pb)
                ioa = wb.ioa(pb)
                rel_area = wb.area / pb.area if pb.area > 0 else 0.0
                ph = pb.height
                pw = pb.width

                is_at_head_top = wb.y1 <= (pb.y1 + 0.08 * ph)
                is_in_cranial_zone = wb.center[1] <= (pb.y1 + 0.22 * ph)
                is_above_neck = wb.y2 <= (pb.y1 + 0.32 * ph)
                is_horizontally_within_head = (
                    wb.center[0] >= (pb.x1 - 15.0)
                    and wb.center[0] <= (pb.x2 + 15.0)
                )
                head_cond = is_at_head_top and is_in_cranial_zone and is_above_neck and is_horizontally_within_head and ioa >= 0.50

                full_body = (iou >= 0.35 and ioa >= 0.60 and rel_area >= 0.28) or (ioa >= 0.75 and rel_area >= 0.38)
                
                is_large_clothing = (
                    ioa >= 0.75
                    and rel_area >= 0.15
                    and (wb.width >= (0.50 * pw) or rel_area >= 0.25)
                )

                reasons.append(
                    f"[vs Person {pb.to_xyxy_int()}: ioa={ioa:.2f}, rel_area={rel_area:.3f}, "
                    f"head_top={is_at_head_top}(wb.y1={wb.y1:.1f} vs {pb.y1+0.08*ph:.1f}), "
                    f"cranial={is_in_cranial_zone}(wb.cy={wb.center[1]:.1f} vs {pb.y1+0.22*ph:.1f}), "
                    f"neck={is_above_neck}(wb.y2={wb.y2:.1f} vs {pb.y1+0.32*ph:.1f}), "
                    f"head_cond={head_cond}, full_body={full_body}, clothing={is_large_clothing}]"
                )
            details.append(f"{wd.class_name}({wd.confidence:.2f})@{wb.to_xyxy_int()}: {' '.join(reasons)}")

        print(f"Frame {f_idx:02d}: Persons={len(p_dets)} RawWaste={len(w_raw)} FiltWaste={len(w_filtered)}")
        for d in details:
            print(f"    Raw: {d}")
        if w_filtered:
            print(f"    --> SURVIVING FILTER: {[(w.class_name, round(w.confidence,2), w.bbox.to_xyxy_int()) for w in w_filtered]}")
    cap.release()
