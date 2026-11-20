import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.common_types import BoundingBox, Detection
from app.tracking.tracker import MultiObjectTrackingService

waste_model = YOLO("models/waste/waste_detection_yolo11m.pt")
person_model = YOLO("models/person/person_detector.pt")

def filter_waste(raw_boxes, frame_w, frame_h, person_boxes):
    # 1. Duplicate suppression (NMS & containment)
    boxes = sorted(raw_boxes, key=lambda x: x['conf'], reverse=True)
    surviving = []
    for b in boxes:
        bbox = b['bbox']
        # Check oversized false detection
        area_frac = bbox.area / (frame_w * frame_h)
        if area_frac >= 0.48 or (bbox.width >= 0.88 * frame_w and bbox.height >= 0.60 * frame_h):
            continue

        # Check duplicate with already accepted box of same class
        is_dup = False
        for s in surviving:
            if s['cls'] == b['cls']:
                iou = bbox.iou(s['bbox'])
                ioa1 = bbox.ioa(s['bbox'])
                ioa2 = s['bbox'].ioa(bbox)
                if iou >= 0.38 or ioa1 >= 0.65 or ioa2 >= 0.65:
                    is_dup = True
                    break
        if is_dup:
            continue

        # Check false human detection (face, head, neck, clothing)
        is_human_fp = False
        for pbox in person_boxes:
            ph = pbox.height
            pw = pbox.width
            ioa = bbox.ioa(pbox)
            iou = bbox.iou(pbox)
            rel_area = bbox.area / pbox.area if pbox.area > 0 else 0

            # Cephalic / facial / neck rejection
            is_at_head_or_face = bbox.y1 <= (pbox.y1 + 0.22 * ph)
            is_in_cranial_facial_zone = bbox.center[1] <= (pbox.y1 + 0.32 * ph)
            is_above_torso = bbox.y2 <= (pbox.y1 + 0.48 * ph)
            is_horizontally_within_head = (
                bbox.center[0] >= (pbox.x1 - 25.0)
                and bbox.center[0] <= (pbox.x2 + 25.0)
            )
            if is_at_head_or_face and is_in_cranial_facial_zone and is_above_torso and is_horizontally_within_head and ioa >= 0.45:
                is_human_fp = True
                break

            # Full / Substantial Body match
            if (iou >= 0.35 and ioa >= 0.60 and rel_area >= 0.28) or (ioa >= 0.75 and rel_area >= 0.38):
                is_human_fp = True
                break

            # Large torso / clothing
            if ioa >= 0.70 and rel_area >= 0.12 and (bbox.width >= 0.42 * pw or rel_area >= 0.20):
                is_human_fp = True
                break

        if not is_human_fp:
            surviving.append(b)

    return surviving

def test_simulation():
    cap = cv2.VideoCapture("data/evidence/evt_20261120_120457_0002_evidence.mp4")
    tracker = MultiObjectTrackingService(high_thresh=0.25, low_thresh=0.10, match_thresh=0.70, track_buffer=30)
    frame_idx = 0

    print("================ SIMULATING CORRECTED PIPELINE ================")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w = frame.shape[:2]

        # Person detection
        p_res = person_model.predict(frame, conf=0.35, imgsz=416, device="cpu", verbose=False)
        person_dets = []
        if p_res and len(p_res[0].boxes) > 0:
            for b in p_res[0].boxes:
                if int(b.cls.cpu().numpy()[0]) == 0:
                    xyxy = b.xyxy.cpu().numpy()[0]
                    person_dets.append(Detection(0, "person", float(b.conf.cpu().numpy()[0]), BoundingBox(*xyxy)))

        # Waste detection (predict down to 0.10 for ByteTrack)
        w_res = waste_model.predict(frame, conf=0.10, imgsz=416, device="cpu", verbose=False)
        raw_boxes = []
        if w_res and len(w_res[0].boxes) > 0:
            for b in w_res[0].boxes:
                xyxy = b.xyxy.cpu().numpy()[0]
                cls = int(b.cls.cpu().numpy()[0])
                conf = float(b.conf.cpu().numpy()[0])
                cname = waste_model.names.get(cls, str(cls))
                raw_boxes.append({'cls': cls, 'cname': cname, 'conf': conf, 'bbox': BoundingBox(*xyxy)})

        p_boxes = [p.bbox for p in person_dets]
        filtered = filter_waste(raw_boxes, w, h, p_boxes)
        waste_dets = [Detection(b['cls'], b['cname'], b['conf'], b['bbox']) for b in filtered]

        t_persons, t_waste = tracker.update(person_dets, waste_dets, frame_idx, float(frame_idx)*0.1)

        print(f"\nFrame {frame_idx:02d}:")
        print(f"  Tracked Persons: {[p.track_id for p in t_persons]}")
        print(f"  Tracked Waste:   {[(w.track_id, w.class_name, f'{w.confidence:.2f}', f'box=[{w.bbox.x1:.0f},{w.bbox.y1:.0f},{w.bbox.x2:.0f},{w.bbox.y2:.0f}]') for w in t_waste]}")

    cap.release()

if __name__ == "__main__":
    test_simulation()
