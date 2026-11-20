import cv2
from pathlib import Path
from ultralytics import YOLO

model = YOLO("models/waste/waste_detection_yolo11m.pt")

for f_num in [1, 2, 3]:
    img_path = f"reports/session_frames/frame_{f_num:02d}.jpg"
    img = cv2.imread(img_path)
    print(f"\n================ Frame {f_num:02d} ================")
    for conf in [0.10, 0.20, 0.25]:
        res = model.predict(img, conf=conf, imgsz=416, device="cpu", verbose=False)
        boxes = res[0].boxes if res else []
        print(f"Conf {conf:.2f} ({len(boxes)} boxes):")
        for b in boxes:
            xyxy = b.xyxy.cpu().numpy()[0]
            c = float(b.conf.cpu().numpy()[0])
            cls = int(b.cls.cpu().numpy()[0])
            cname = model.names.get(cls, str(cls))
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            print(f"  {cname} (conf={c:.3f}) @ [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}] w={w:.1f} h={h:.1f}")
