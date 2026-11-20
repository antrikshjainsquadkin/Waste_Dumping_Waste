"""Check zone bounds against left side detections."""

import cv2
from pathlib import Path

img = cv2.imread("reports/unknown_frames/evt_20261119_185408_0005_evidence_f01.jpg")
print("Width:", img.shape[1], "Height:", img.shape[0])
print("Bin pixel coords: x in [448, 608], y in [264, 456]")
print("Left side detection (41, 174, 111, 369) is in open_ground_default!")
