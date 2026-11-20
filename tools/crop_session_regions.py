import cv2
import numpy as np
from pathlib import Path

# Load frame_01, frame_04, frame_08, frame_10
in_dir = Path("reports/session_frames")
out_dir = Path("reports/annotated_session")
out_dir.mkdir(parents=True, exist_ok=True)

# For each frame, save a crop of:
# 1. Left side [0, 150, 250, 480]
# 2. Person [200, 120, 550, 480]
# 3. Ground / table [280, 380, 450, 480]
for idx in [1, 2, 4, 6, 8, 10]:
    fpath = in_dir / f"frame_{idx:02d}.jpg"
    img = cv2.imread(str(fpath))
    if img is None:
        continue
    h, w = img.shape[:2]
    # Crop person region
    p_crop = img[100:480, 150:560]
    cv2.imwrite(str(out_dir / f"person_crop_{idx:02d}.jpg"), p_crop)
    # Crop left region
    l_crop = img[150:480, 0:250]
    cv2.imwrite(str(out_dir / f"left_crop_{idx:02d}.jpg"), l_crop)
    print(f"Processed frame {idx}")

print("Done cropping regions.")
