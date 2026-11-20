"""Extract frames from evidence clips for inspection."""

import sys
from pathlib import Path
import cv2

out_dir = Path("reports/unknown_frames")
out_dir.mkdir(parents=True, exist_ok=True)

for clip_name in [
    "evt_20261119_185229_0001_evidence.mp4",
    "evt_20261119_185408_0005_evidence.mp4",
    "evt_20261120_110246_0001_evidence.mp4",
]:
    p = Path("data/evidence") / clip_name
    if not p.is_file():
        continue
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(clip_name, "has", n, "frames")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        cv2.imwrite(str(out_dir / f"{clip_name[:-4]}_f{idx:02d}.jpg"), frame)
    cap.release()
print("Extracted successfully.")
