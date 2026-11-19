"""Dedicated Waste Detector wrapping the official Hugging Face YOLO11m checkpoint."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from app.common_types import BoundingBox, Detection

logger = logging.getLogger(__name__)

EXPECTED_WASTE_SHA256 = "9f2d19a7ce386c6dbb95a725e6197f7b6fb699cd601bbeca9407034396f2ad6a"
EXPECTED_CLASSES = {0: "organic", 1: "recyclable", 2: "residual"}


class WasteDetector:
    """YOLO11m waste detector specialized for organic, recyclable, and residual waste."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        image_size: int = 416,
        device: str = "cpu",
        verify_sha256: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.device = str(device)
        self.verify_sha256 = verify_sha256
        self.model: Optional[YOLO] = None
        self.class_names = EXPECTED_CLASSES.copy()
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO11m waste model and verify checksum & taxonomy."""
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Waste detection model not found at: {self.model_path}")

        if self.verify_sha256:
            logger.info("Verifying SHA-256 for waste model checkpoint: %s", self.model_path)
            h = hashlib.sha256()
            with open(self.model_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            calc_sha = h.hexdigest()
            if calc_sha != EXPECTED_WASTE_SHA256:
                raise ValueError(
                    f"Waste model checksum mismatch! Expected {EXPECTED_WASTE_SHA256}, got {calc_sha}"
                )
            logger.info("Waste model SHA-256 verified successfully.")

        logger.info("Loading YOLO11m waste model on device=%s...", self.device)
        self.model = YOLO(str(self.model_path))
        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            self.class_names = self.model.names
        logger.info("WasteDetector ready. Native classes: %s", self.class_names)

    def detect(
        self,
        frame: np.ndarray,
        person_boxes: Optional[List[BoundingBox]] = None,
    ) -> List[Detection]:
        """Run waste inference on frame and apply non-waste rejection filter against people."""
        if self.model is None or frame is None or frame.size == 0:
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error("WasteDetector prediction failed: %s", e)
            return []

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            xyxy = box.xyxy.cpu().numpy()[0]
            bbox = BoundingBox(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
            )
            cname = self.class_names.get(cls_id, f"waste_{cls_id}")

            # 3-Phase Non-Waste False Detection Rejection
            if person_boxes and self._is_false_human_detection(bbox, conf, person_boxes):
                continue

            detections.append(Detection(class_id=cls_id, class_name=cname, confidence=conf, bbox=bbox))

        return detections

    def _is_false_human_detection(
        self,
        waste_box: BoundingBox,
        confidence: float,
        person_boxes: List[BoundingBox],
    ) -> bool:
        """Reject false positive waste detections caused by person head/hair, full body, or large clothing.

        Preserves genuine hand-held waste objects located in the torso/hand carrying volume.
        """
        for pbox in person_boxes:
            iou = waste_box.iou(pbox)
            ioa = waste_box.ioa(pbox)
            rel_area = waste_box.area / pbox.area if pbox.area > 0 else 0.0
            ph = pbox.height
            pw = pbox.width

            # 1. Head / Hair False Positive Rejection:
            # Anatomically, the top of pbox (pbox.y1) is the crown of the head.
            # Detections starting at/near the scalp (pbox.y1) and centered in the cranial zone
            # (<= 22% of person height) are hair/facial features misclassified as recyclable/residual.
            is_at_head_top = waste_box.y1 <= (pbox.y1 + 0.08 * ph)
            is_in_cranial_zone = waste_box.center[1] <= (pbox.y1 + 0.22 * ph)
            is_above_neck = waste_box.y2 <= (pbox.y1 + 0.32 * ph)
            is_horizontally_within_head = (
                waste_box.center[0] >= (pbox.x1 - 15.0)
                and waste_box.center[0] <= (pbox.x2 + 15.0)
            )
            if is_at_head_top and is_in_cranial_zone and is_above_neck and is_horizontally_within_head and ioa >= 0.50:
                logger.debug("Rejecting head/hair false positive: box=%s, conf=%.3f", waste_box, confidence)
                return True

            # 2. Full / Substantial Body match: waste box almost coincides with person box
            if (iou >= 0.35 and ioa >= 0.60 and rel_area >= 0.28) or (ioa >= 0.75 and rel_area >= 0.38):
                logger.debug("Rejecting full body false match: box=%s, iou=%.2f, rel_area=%.2f", waste_box, iou, rel_area)
                return True

            # 3. Large Torso / Clothing match:
            # Major clothing items (entire shirt, coat, pants, torso patch) spanning significant portion of person area
            is_large_clothing = (
                ioa >= 0.75
                and rel_area >= 0.15
                and (waste_box.width >= (0.50 * pw) or rel_area >= 0.25)
            )
            if is_large_clothing:
                logger.debug("Rejecting large torso/clothing detection: box=%s, rel_area=%.2f", waste_box, rel_area)
                return True

        # Genuine hand-held items (bottles, bags, cups, wrappers) in the carrying volume
        # (rel_area < 0.20, below cranial zone) are PRESERVED regardless of ioa with person.
        return False
