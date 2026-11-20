import sys
from pathlib import Path
import cv2
import numpy as np
import math
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.common_types import BoundingBox, Detection, TrackedObject
from app.tracking.tracker import MultiObjectTrackingService, SingleCategoryTracker

waste_model = YOLO("models/waste/waste_detection_yolo11m.pt")
person_model = YOLO("models/person/person_detector.pt")

class EnhancedWasteTracker(SingleCategoryTracker):
    """Waste tracker with drop/occlusion trajectory reconnection."""

    def update(self, detections, frame_idx, timestamp):
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        active_track_ids = [tid for tid, lost in self._lost_frames.items() if lost == 0]
        lost_track_ids = [tid for tid, lost in self._lost_frames.items() if 0 < lost <= self.max_lost_frames]

        unmatched_active = set(active_track_ids)
        unmatched_high = list(range(len(high_dets)))

        # 1. Stage 1: Active tracks with high-confidence detections via IoU matching
        if active_track_ids and high_dets:
            matches, u_tracks, u_high = self._match(
                [self._active_tracks[tid].bbox for tid in active_track_ids],
                [d.bbox for d in high_dets],
                threshold=self.match_thresh,
            )
            for t_idx, d_idx in matches:
                tid = active_track_ids[t_idx]
                det = high_dets[d_idx]
                self._active_tracks[tid].update(det.bbox, det.confidence, frame_idx, timestamp)
                self._lost_frames[tid] = 0
                unmatched_active.discard(tid)
            unmatched_high = u_high

        # 2. Stage 2: Remaining active tracks with low-confidence detections
        if unmatched_active and low_dets:
            rem_active = list(unmatched_active)
            matches_low, _, _ = self._match(
                [self._active_tracks[tid].bbox for tid in rem_active],
                [d.bbox for d in low_dets],
                threshold=max(0.30, self.match_thresh - 0.25),
            )
            for t_idx, d_idx in matches_low:
                tid = rem_active[t_idx]
                det = low_dets[d_idx]
                self._active_tracks[tid].update(det.bbox, det.confidence, frame_idx, timestamp)
                self._lost_frames[tid] = 0
                unmatched_active.discard(tid)

        # 3. Stage 3: Trajectory continuity for dropped / released objects or recently lost tracks
        # Candidate tracks: unmatched active tracks + recently lost tracks (lost <= 20 frames)
        candidate_tracks = list(unmatched_active) + [tid for tid in lost_track_ids if self._lost_frames.get(tid, 999) <= 20]
        unmatched_high_final = []

        for d_idx in unmatched_high:
            det = high_dets[d_idx]
            best_tid = None
            best_dist = float('inf')

            for tid in candidate_tracks:
                track = self._active_tracks[tid]
                if track.class_name != det.class_name:
                    continue
                dist = track.bbox.distance_to(det.bbox)
                dy = det.bbox.center[1] - track.bbox.center[1]
                # Drop release or displacement: dist <= 160 px, downward or nearby (dy >= -30)
                if dist <= 160.0 and dy >= -30.0 and dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                self._active_tracks[best_tid].update(det.bbox, det.confidence, frame_idx, timestamp)
                self._lost_frames[best_tid] = 0
                unmatched_active.discard(best_tid)
                if best_tid in lost_track_ids:
                    lost_track_ids.remove(best_tid)
                candidate_tracks.remove(best_tid)
                print(f"  [CONTINUITY] Track Waste_{best_tid} maintained across drop/motion to {det.bbox.center} (dist={best_dist:.1f}px)")
            else:
                unmatched_high_final.append(d_idx)

        # 4. Increment lost frames and prune dead tracks
        for tid in unmatched_active:
            self._lost_frames[tid] = self._lost_frames.get(tid, 0) + 1
        for tid in lost_track_ids:
            self._lost_frames[tid] = self._lost_frames.get(tid, 0) + 1

        dead_ids = [tid for tid, lost in self._lost_frames.items() if lost > self.max_lost_frames]
        for tid in dead_ids:
            self._active_tracks.pop(tid, None)
            self._lost_frames.pop(tid, None)

        # 5. Initialize new tracks for genuinely new objects
        for d_idx in unmatched_high_final:
            det = high_dets[d_idx]
            tid = self._next_id
            self._next_id += 1
            track_obj = TrackedObject(
                track_id=f"{self.prefix}_{tid}",
                raw_id=tid,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                last_frame_idx=frame_idx,
                last_timestamp=timestamp,
                history=[(frame_idx, timestamp, det.bbox)],
            )
            self._active_tracks[tid] = track_obj
            self._lost_frames[tid] = 0

        return [t for tid, t in self._active_tracks.items() if self._lost_frames.get(tid, 0) == 0]

def filter_waste(raw_boxes, frame_w, frame_h, person_boxes):
    boxes = sorted(raw_boxes, key=lambda x: x['conf'], reverse=True)
    surviving = []
    for b in boxes:
        bbox = b['bbox']
        area_frac = bbox.area / (frame_w * frame_h)
        if area_frac >= 0.48 or (bbox.width >= 0.88 * frame_w and bbox.height >= 0.60 * frame_h):
            continue

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

        is_human_fp = False
        for pbox in person_boxes:
            ph = pbox.height
            pw = pbox.width
            ioa = bbox.ioa(pbox)
            iou = bbox.iou(pbox)
            rel_area = bbox.area / pbox.area if pbox.area > 0 else 0

            # Cephalic / facial / neck rejection (covers up to 48% height)
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

            if (iou >= 0.35 and ioa >= 0.60 and rel_area >= 0.28) or (ioa >= 0.75 and rel_area >= 0.38):
                is_human_fp = True
                break

            if ioa >= 0.70 and rel_area >= 0.12 and (bbox.width >= 0.42 * pw or rel_area >= 0.20):
                is_human_fp = True
                break

        if not is_human_fp:
            surviving.append(b)
    return surviving

def run():
    cap = cv2.VideoCapture("data/evidence/evt_20261120_120457_0002_evidence.mp4")
    tracker = EnhancedWasteTracker(namespace_prefix="Waste", high_thresh=0.25, low_thresh=0.10, match_thresh=0.70, max_lost_frames=30)
    frame_idx = 0
    print("Testing Enhanced Track Reconnection on real clip:")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w = frame.shape[:2]

        p_res = person_model.predict(frame, conf=0.35, imgsz=416, device="cpu", verbose=False)
        person_dets = []
        if p_res and len(p_res[0].boxes) > 0:
            for b in p_res[0].boxes:
                if int(b.cls.cpu().numpy()[0]) == 0:
                    xyxy = b.xyxy.cpu().numpy()[0]
                    person_dets.append(Detection(0, "person", float(b.conf.cpu().numpy()[0]), BoundingBox(*xyxy)))

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

        t_waste = tracker.update(waste_dets, frame_idx, float(frame_idx)*0.1)
        print(f"Frame {frame_idx:02d}: {[ (w.track_id, w.class_name, f'conf={w.confidence:.2f}', f'box=[{w.bbox.x1:.0f},{w.bbox.y1:.0f},{w.bbox.x2:.0f},{w.bbox.y2:.0f}]') for w in t_waste ]}")

    cap.release()

if __name__ == "__main__":
    run()
