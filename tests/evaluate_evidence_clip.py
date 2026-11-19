"""Inspect detections across frames in real captured evidence video."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.detection.person_detector import PersonDetector
from app.detection.waste_detector import WasteDetector


def evaluate():
    video_path = PROJECT_ROOT / "data" / "evidence" / "evt_20261119_104123_0001_evidence.mp4"
    if not video_path.is_file():
        print(f"File not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    p_det = PersonDetector(PROJECT_ROOT / "models" / "person" / "person_detector.pt")
    w_det = WasteDetector(PROJECT_ROOT / "models" / "waste" / "waste_detection_yolo11m.pt", verify_sha256=False)

    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f_idx += 1
        persons = p_det.detect(frame)
        p_boxes = [p.bbox for p in persons]
        waste = w_det.detect(frame, person_boxes=p_boxes)
        p_info = [f"conf={p.confidence:.2f}, box=({int(p.bbox.x1)},{int(p.bbox.y1)},{int(p.bbox.x2)},{int(p.bbox.y2)})" for p in persons]
        w_info = [f"{w.class_name}({w.confidence:.2f}) box=({int(w.bbox.x1)},{int(w.bbox.y1)},{int(w.bbox.x2)},{int(w.bbox.y2)})" for w in waste]
        print(f"Frame {f_idx}: People={len(persons)} {p_info} | Waste={len(waste)} {w_info}")
    cap.release()


if __name__ == "__main__":
    evaluate()
