"""Face Recognition Adapter wrapping InsightFace and enrolled identity database."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

from app.common_types import ConfirmedLitterEvent, RecognitionStatus

logger = logging.getLogger(__name__)


class FaceRecognitionAdapter:
    """Event-triggered facial recognition adapter for offender identification."""

    def __init__(
        self,
        faces_json_path: str | Path,
        model_name: str = "buffalo_s",
        similarity_threshold: float = 0.50,
        ambiguity_margin: float = 0.03,
        min_det_score: float = 0.35,
        min_face_size_px: int = 30,
        min_sharpness: float = 12.0,
    ) -> None:
        self.faces_json_path = Path(faces_json_path)
        self.model_name = str(model_name)
        self.sim_threshold = float(similarity_threshold)
        self.ambiguity_margin = float(ambiguity_margin)
        self.min_det_score = float(min_det_score)
        self.min_face_size = int(min_face_size_px)
        self.min_sharpness = float(min_sharpness)

        self._registry: Dict[str, np.ndarray] = {}
        self._app: Optional[FaceAnalysis] = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """Initialize InsightFace analysis model and load enrolled identities."""
        self._load_registry()

        if not HAS_INSIGHTFACE:
            logger.warning("insightface package is not installed. Face recognition disabled.")
            return False

        try:
            logger.info("Initializing InsightFace with model '%s' on CPU...", self.model_name)
            self._app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            self._is_initialized = True
            logger.info("FaceRecognitionAdapter initialized successfully.")
            return True
        except Exception as e:
            logger.error("Failed to initialize FaceAnalysis: %s", e)
            self._is_initialized = False
            return False

    def _load_registry(self) -> None:
        """Load enrolled face embeddings from faces.json."""
        if not self.faces_json_path.is_file():
            logger.warning("faces.json not found at %s. Running with empty registry.", self.faces_json_path)
            return

        try:
            with open(self.faces_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._registry.clear()
            for name, val in data.items():
                if isinstance(val, dict) and "embedding" in val:
                    vec = np.array(val["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        self._registry[name] = vec / norm
                elif isinstance(val, list):
                    vec = np.array(val, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        self._registry[name] = vec / norm

            logger.info("Loaded %d enrolled identities from %s: %s", len(self._registry), self.faces_json_path, list(self._registry.keys()))
        except Exception as e:
            logger.error("Failed to parse faces.json: %s", e)

    @property
    def is_available(self) -> bool:
        return self._is_initialized and self._app is not None

    def process_event(
        self,
        event: ConfirmedLitterEvent,
        frames: List[Tuple[float, np.ndarray, int]],
        person_box: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[str, float, RecognitionStatus, Optional[np.ndarray]]:
        """Identify offender from event video window. Runs ONLY on confirmed event."""
        if not frames:
            return "Unknown", 0.0, RecognitionStatus.NO_FACE, None

        if not self.is_available:
            return "Unknown", 0.0, RecognitionStatus.SKIPPED, None

        best_face_crop: Optional[np.ndarray] = None
        best_face_score = 0.0
        best_embedding: Optional[np.ndarray] = None

        # Sample frames chronologically (sample every 2-3 frames to save CPU)
        sampled_frames = frames[::2] if len(frames) > 10 else frames

        for _, frame, _ in sampled_frames:
            # Crop to person box if available, otherwise analyze whole frame
            search_img = frame
            offset_x, offset_y = 0, 0
            if person_box is not None:
                px1, py1, px2, py2 = [int(v) for v in person_box]
                h_f, w_f = frame.shape[:2]
                px1, py1 = max(0, px1 - 20), max(0, py1 - 20)
                px2, py2 = min(w_f, px2 + 20), min(h_f, py2 + 20)
                if (px2 - px1) > self.min_face_size and (py2 - py1) > self.min_face_size:
                    search_img = frame[py1:py2, px1:px2]
                    offset_x, offset_y = px1, py1

            try:
                faces = self._app.get(search_img)
            except Exception as e:
                logger.error("Face detection failed in event frame: %s", e)
                continue

            for face in faces:
                if face.det_score < self.min_det_score:
                    continue

                bbox = face.bbox.astype(int)
                fw = bbox[2] - bbox[0]
                fh = bbox[3] - bbox[1]
                if fw < self.min_face_size or fh < self.min_face_size:
                    continue

                # Crop face and evaluate sharpness (Laplacian variance)
                fx1 = max(0, bbox[0] + offset_x)
                fy1 = max(0, bbox[1] + offset_y)
                fx2 = min(frame.shape[1], bbox[2] + offset_x)
                fy2 = min(frame.shape[0], bbox[3] + offset_y)
                face_crop = frame[fy1:fy2, fx1:fx2]
                if face_crop.size == 0:
                    continue

                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sharpness < self.min_sharpness:
                    continue

                overall_quality = face.det_score * 0.5 + min(100.0, sharpness) * 0.005
                if overall_quality > best_face_score and face.embedding is not None:
                    best_face_score = overall_quality
                    best_face_crop = face_crop.copy()
                    emb = face.embedding.copy()
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        best_embedding = emb / norm

        if best_embedding is None:
            return "Unknown", 0.0, RecognitionStatus.NO_FACE, best_face_crop

        # Match against enrolled registry
        if not self._registry:
            return "Unknown", 0.0, RecognitionStatus.UNKNOWN, best_face_crop

        matches = []
        for name, registered_vec in self._registry.items():
            sim = float(np.dot(best_embedding, registered_vec))
            matches.append((name, sim))

        matches.sort(key=lambda x: x[1], reverse=True)
        top_name, top_sim = matches[0]

        if top_sim < self.sim_threshold:
            return "Unknown", top_sim, RecognitionStatus.UNKNOWN, best_face_crop

        # Check ambiguity margin if second candidate exists
        if len(matches) > 1:
            second_name, second_sim = matches[1]
            if (top_sim - second_sim) < self.ambiguity_margin:
                logger.info("Ambiguous face match between %s (%.3f) and %s (%.3f)", top_name, top_sim, second_name, second_sim)
                return "Unknown", top_sim, RecognitionStatus.UNKNOWN, best_face_crop

        return top_name, top_sim, RecognitionStatus.IDENTIFIED, best_face_crop
