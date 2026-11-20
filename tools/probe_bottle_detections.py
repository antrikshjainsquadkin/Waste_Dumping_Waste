import cv2
from pathlib import Path
from ultralytics import YOLO

model = YOLO("models/waste/waste_detection_yolo11m.pt")

img_path = "reports/session_frames/frame_04.jpg"
img = cv2.imread(img_path)

print(f"--- Full Frame 04 Inference with multiple conf thresholds ---")
for conf in [0.10, 0.15, 0.20, 0.25]:
    res = model.predict(img, conf=conf, imgsz=416, device="cpu", verbose=False)
    boxes = res[0].boxes if res else []
    print(f"\nConf threshold {conf:.2f}: {len(boxes)} boxes")
    for b in boxes:
        xyxy = b.xyxy.cpu().numpy()[0]
        c = float(b.conf.cpu().numpy()[0])
        cls = int(b.cls.cpu().numpy()[0])
        cname = model.names.get(cls, str(cls))
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        print(f"  {cname} (conf={c:.3f}) @ [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}] w={w:.1f} h={h:.1f}")

# Also test on person crop
p_crop = img[100:480, 150:560]
print(f"\n--- Person Crop Frame 04 Inference ---")
for conf in [0.10, 0.15, 0.20, 0.25]:
    res = model.predict(p_crop, conf=conf, imgsz=416, device="cpu", verbose=False)
    boxes = res[0].boxes if res else []
    print(f"\nPerson Crop Conf threshold {conf:.2f}: {len(boxes)} boxes")
    for b in boxes:
        xyxy = b.xyxy.cpu().numpy()[0]
        c = float(b.conf.cpu().numpy()[0])
        cls = int(b.cls.cpu().numpy()[0])
        cname = model.names.get(cls, str(cls))
        print(f"  {cname} (conf={c:.3f}) @ crop [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}] -> frame [{xyxy[0]+150:.1f}, {xyxy[1]+100:.1f}, {xyxy[2]+150:.1f}, {xyxy[3]+100:.1f}]")
