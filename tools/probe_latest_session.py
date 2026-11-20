import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.detection.waste_detector import WasteDetector
from app.detection.person_detector import PersonDetector
from app.common_types import BoundingBox

def inspect_clip(clip_path: str):
    print(f"\n==================================================")
    print(f"Inspecting {clip_path}")
    print(f"==================================================")
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"Failed to open {clip_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {w}x{h}, FPS: {fps}, Total Frames: {frame_count}")

    waste_detector = WasteDetector("models/waste/waste_detection_yolo11m.pt", verify_sha256=False)
    person_detector = PersonDetector("models/person/person_detector.pt")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1. Raw YOLO person predictions
        person_results = person_detector.model.predict(frame, conf=0.35, imgsz=416, device="cpu", verbose=False)
        person_boxes = []
        if person_results and len(person_results[0].boxes) > 0:
            for b in person_results[0].boxes:
                xyxy = b.xyxy.cpu().numpy()[0]
                conf = float(b.conf.cpu().numpy()[0])
                cls = int(b.cls.cpu().numpy()[0])
                if cls == 0:
                    person_boxes.append((conf, xyxy))

        # 2. Raw YOLO waste predictions
        waste_results = waste_detector.model.predict(frame, conf=0.25, imgsz=416, device="cpu", verbose=False)
        raw_waste = []
        if waste_results and len(waste_results[0].boxes) > 0:
            for b in waste_results[0].boxes:
                xyxy = b.xyxy.cpu().numpy()[0]
                conf = float(b.conf.cpu().numpy()[0])
                cls = int(b.cls.cpu().numpy()[0])
                cname = waste_detector.class_names.get(cls, str(cls))
                raw_waste.append((cname, conf, xyxy))

        # 3. Post-filtered waste predictions
        p_bboxes = [BoundingBox(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])) for conf, xyxy in person_boxes]
        filtered_waste = waste_detector.detect(frame, person_boxes=p_bboxes)

        print(f"\n--- Frame {frame_idx:02d} ---")
        print(f"Persons ({len(person_boxes)}):")
        for conf, xyxy in person_boxes:
            print(f"  Person conf={conf:.2f} box=[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}] w={xyxy[2]-xyxy[0]:.1f} h={xyxy[3]-xyxy[1]:.1f}")

        print(f"Raw Waste ({len(raw_waste)}):")
        for cname, conf, xyxy in raw_waste:
            w_box = xyxy[2] - xyxy[0]
            h_box = xyxy[3] - xyxy[1]
            area_frac = (w_box * h_box) / (w * h)
            print(f"  {cname} conf={conf:.2f} box=[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}] w={w_box:.1f} h={h_box:.1f} (area {area_frac*100:.1f}% frame)")

        print(f"Filtered Waste ({len(filtered_waste)}):")
        for fw in filtered_waste:
            print(f"  {fw.class_name} conf={fw.confidence:.2f} box=[{fw.bbox.x1:.1f}, {fw.bbox.y1:.1f}, {fw.bbox.x2:.1f}, {fw.bbox.y2:.1f}]")

    cap.release()

if __name__ == "__main__":
    clip = "data/evidence/evt_20261120_120457_0002_evidence.mp4"
    if len(sys.argv) > 1:
        clip = sys.argv[1]
    inspect_clip(clip)
