"""Dedicated Person Detector using YOLO11 COCO pretrained weights."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from ultralytics import YOLO

from app.common_types import BoundingBox, Detection

logger = logging.getLogger(__name__)


class PersonDetector:
    """YOLO11 detector strictly dedicated to detecting people (COCO class 0)."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.35,
        image_size: int = 416,
        device: str = "cpu",
        person_class_id: int = 0,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.device = str(device)
        self.person_class_id = int(person_class_id)
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO person detector."""
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Person detector checkpoint not found at: {self.model_path}")

        logger.info("Loading PersonDetector model from %s on %s...", self.model_path, self.device)
        self.model = YOLO(str(self.model_path))
        cname = self.model.names.get(self.person_class_id, "unknown")
        logger.info("PersonDetector loaded. Target class ID %d is '%s'", self.person_class_id, cname)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect people in frame, returning only person class detections."""
        if self.model is None or frame is None or frame.size == 0:
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                classes=[self.person_class_id],
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error("PersonDetector prediction failed: %s", e)
            return []

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            if cls_id != self.person_class_id:
                continue
            conf = float(box.conf.cpu().numpy()[0])
            xyxy = box.xyxy.cpu().numpy()[0]
            bbox = BoundingBox(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
            )
            detections.append(Detection(class_id=cls_id, class_name="person", confidence=conf, bbox=bbox))

        return detections
