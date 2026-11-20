"""Inspect what is visually at the detected waste boxes."""

from pathlib import Path
import cv2
import numpy as np

img = cv2.imread("reports/unknown_frames/evt_20261119_185408_0005_evidence_f01.jpg")
print("Image shape:", img.shape)

# Let's crop the detection (41, 174, 111, 369)
# x1=41, y1=174, x2=111, y2=369
crop = img[174:369, 41:111]
cv2.imwrite("reports/unknown_frames/crop_waste_41_174.jpg", crop)
print("Crop 41_174 saved, shape:", crop.shape, "mean BGR:", crop.mean(axis=(0,1)))

# Let's also check evt_20261119_185229_0001_evidence_f01
img1 = cv2.imread("reports/unknown_frames/evt_20261119_185229_0001_evidence_f01.jpg")
crop_head = img1[322:356, 275:339]
cv2.imwrite("reports/unknown_frames/crop_head_275_322.jpg", crop_head)
print("Crop head saved, shape:", crop_head.shape, "mean BGR:", crop_head.mean(axis=(0,1)))

# Check person crops and face snapshots
face1 = cv2.imread("data/evidence/evt_20261119_185229_0001_face.jpg")
print("Face 185229 shape:", face1.shape if face1 is not None else None)
face_antriksh = cv2.imread("data/evidence/evt_20261120_110246_0001_face.jpg")
print("Face antriksh shape:", face_antriksh.shape if face_antriksh is not None else None)
