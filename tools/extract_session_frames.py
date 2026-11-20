import cv2
from pathlib import Path

clip = "data/evidence/evt_20261120_120457_0002_evidence.mp4"
out_dir = Path("reports/session_frames")
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(clip)
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    idx += 1
    out_path = out_dir / f"frame_{idx:02d}.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"Wrote frame {idx} to {out_path}")

cap.release()
print(f"Extracted {idx} frames.")
