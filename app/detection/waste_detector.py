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

        # Predict down to 0.10 to feed ByteTrack two-stage matching for occluded items
        predict_conf = min(self.confidence_threshold, 0.10)
        try:
            results = self.model.predict(
                source=frame,
                conf=predict_conf,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error("WasteDetector prediction failed: %s", e)
            return []

        if not results or len(results) == 0:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        fh, fw = frame.shape[:2]
        frame_area = float(fw * fh) if (fw > 0 and fh > 0) else 1.0

        raw_candidates: List[Detection] = []
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

            # 1. Oversized False Bounding Box Rejection:
            # Candidate waste boxes covering >= 48% of the entire frame area or spanning >= 88% width
            # and >= 60% height represent room/wall/background false alarms, not discrete waste items.
            area_frac = bbox.area / frame_area
            if area_frac >= 0.48 or (bbox.width >= 0.88 * fw and bbox.height >= 0.60 * fh):
                logger.debug("Rejecting oversized frame-level false detection: box=%s, area_frac=%.2f", bbox, area_frac)
                continue

            # 2. Multi-Phase False Human Detection Rejection
            if person_boxes and self._is_false_human_detection(bbox, conf, person_boxes):
                continue

            raw_candidates.append(Detection(class_id=cls_id, class_name=cname, confidence=conf, bbox=bbox))

        # 3. Class-Aware Duplicate & Containment Suppression:
        # Avoid duplicate tracks for the same physical object while preserving distinct adjacent items
        raw_candidates.sort(key=lambda d: d.confidence, reverse=True)
        detections: List[Detection] = []
        for cand in raw_candidates:
            is_dup = False
            for accepted in detections:
                if accepted.class_id == cand.class_id:
                    iou = cand.bbox.iou(accepted.bbox)
                    ioa1 = cand.bbox.ioa(accepted.bbox)
                    ioa2 = accepted.bbox.ioa(cand.bbox)
                    if iou >= 0.38 or ioa1 >= 0.65 or ioa2 >= 0.65:
                        is_dup = True
                        break
            if not is_dup:
                detections.append(cand)

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

            # 1. Head / Hair / Face / Neck False Positive Rejection:
            # Anatomically, the upper 48% of the person bounding box corresponds to
            # head crown, hair, forehead, eyes, nose, cheeks, mouth, jaw/chin, and neck/collar.
            # Detections whose center is in this cranial/facial region (<= 32% of person height),
            # with top starting in the head/face zone (<= 22% of height),
            # bottom not extending into waist/hand carrying region (<= 48% of height),
            # horizontally aligned with the head/upper body, and overlapping the person (ioa >= 0.45)
            # are facial features, hair, or neck misclassified as recyclable/residual.
            is_at_head_or_face = waste_box.y1 <= (pbox.y1 + 0.22 * ph)
            is_in_cranial_facial_zone = waste_box.center[1] <= (pbox.y1 + 0.32 * ph)
            is_above_torso = waste_box.y2 <= (pbox.y1 + 0.48 * ph)
            is_horizontally_within_head = (
                waste_box.center[0] >= (pbox.x1 - 25.0)
                and waste_box.center[0] <= (pbox.x2 + 25.0)
            )
            if (
                is_at_head_or_face
                and is_in_cranial_facial_zone
                and is_above_torso
                and is_horizontally_within_head
                and ioa >= 0.45
            ):
                logger.debug("Rejecting head/face/hair false positive: box=%s, conf=%.3f", waste_box, confidence)
                return True

            # 2. Full / Substantial Body match: waste box almost coincides with person box
            if (iou >= 0.35 and ioa >= 0.60 and rel_area >= 0.28) or (ioa >= 0.75 and rel_area >= 0.38):
                logger.debug("Rejecting full body false match: box=%s, iou=%.2f, rel_area=%.2f", waste_box, iou, rel_area)
                return True

            # 3. Large Torso / Clothing match:
            # Major clothing items (entire shirt, coat, pants, torso patch) spanning significant portion of person area
            is_large_clothing = (
                ioa >= 0.70
                and rel_area >= 0.12
                and (waste_box.width >= (0.42 * pw) or rel_area >= 0.20)
            )
            if is_large_clothing:
                logger.debug("Rejecting large torso/clothing detection: box=%s, rel_area=%.2f", waste_box, rel_area)
                return True

        # Genuine hand-held items (bottles, bags, cups, wrappers) in the carrying volume
        # (rel_area < 0.20, below cranial zone) are PRESERVED regardless of ioa with person.
        return False
