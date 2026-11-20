"""Inspect scene and annotate frame."""

import cv2
from pathlib import Path

img = cv2.imread("reports/unknown_frames/evt_20261119_185408_0005_evidence_f01.jpg")
if img is not None:
    # Draw boxes
    cv2.rectangle(img, (41, 174), (111, 369), (0, 0, 255), 2)
    cv2.putText(img, "Waste_12 (0.60)", (41, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.rectangle(img, (226, 48), (640, 463), (255, 0, 0), 2)
    cv2.putText(img, "Person_2", (226, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite("reports/unknown_frames/annotated_185408_f01.jpg", img)
    print("Saved annotated_185408_f01.jpg")

img2 = cv2.imread("reports/unknown_frames/evt_20261119_185229_0001_evidence_f01.jpg")
if img2 is not None:
    # Waste (275, 322, 339, 356) and Person (280, 291, 575, 480)
    cv2.rectangle(img2, (275, 322), (339, 356), (0, 0, 255), 2)
    cv2.putText(img2, "Waste (0.26)", (275, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.rectangle(img2, (280, 291), (575, 480), (255, 0, 0), 2)
    cv2.putText(img2, "Person_2", (280, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite("reports/unknown_frames/annotated_185229_f01.jpg", img2)
    print("Saved annotated_185229_f01.jpg")
