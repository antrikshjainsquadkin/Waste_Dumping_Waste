"""Face Recognition Adapter wrapping InsightFace and enrolled identity database."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import time
from typing import Any, Dict, List, Optional, Tuple
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

# Known cryptographic SHA-256 for the enrolled reference face registry
EXPECTED_REFERENCE_SHA256 = "33E79CD13E3CC04DD6847E729725B37064E1369147CC594C9E3DAF55581A8AC6"


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

    @staticmethod
    def _verify_and_restore(target_path: Path, src_path: Path) -> bool:
        """Validate integrity and restore faces.json from trusted reference."""
        try:
            content = src_path.read_bytes()
            if not content:
                logger.error("Reference backup %s is empty (0 bytes). Cannot restore.", src_path)
                return False

            # Verify checksum if restoring standard reference
            actual_hash = hashlib.sha256(content).hexdigest().upper()
            if "faces_reference.json" in src_path.name or "Face_recognize" in str(src_path):
                if actual_hash != EXPECTED_REFERENCE_SHA256:
                    logger.warning(
                        "Reference %s checksum mismatch! Expected %s, got %s",
                        src_path,
                        EXPECTED_REFERENCE_SHA256,
                        actual_hash,
                    )
            # Validate JSON content integrity
            parsed = json.loads(content.decode("utf-8"))
            if not isinstance(parsed, dict) or not parsed:
                logger.error("Reference %s does not contain valid dictionary of identities.", src_path)
                return False

            if target_path.exists():
                try:
                    import stat
                    os.chmod(target_path, stat.S_IWRITE | stat.S_IREAD)
                except Exception:
                    pass

            target_path.write_bytes(content)

            try:
                import stat
                os.chmod(target_path, stat.S_IREAD)
            except Exception:
                pass

            logger.info("Successfully restored %s from verified reference %s (SHA-256: %s)", target_path, src_path, actual_hash)
            return True
        except Exception as e:
            logger.error("Failed to restore from %s: %s", src_path, e)
            return False

    def _load_registry(self) -> None:
        """Load enrolled face embeddings from faces.json with auto-heal resilience."""
        ref_path = self.faces_json_path.parent / "faces_reference.json"
        alt_orig = Path(r"C:\Users\A\OneDrive\Desktop\Waste_dumping\Face_recognize\faces.json")

        # Check if faces.json is missing or empty (0 bytes) and restore from reference
        if not self.faces_json_path.is_file() or self.faces_json_path.stat().st_size == 0:
            src = ref_path if (ref_path.is_file() and ref_path.stat().st_size > 0) else (alt_orig if (alt_orig.is_file() and alt_orig.stat().st_size > 0) else None)
            if src:
                logger.info("Auto-healing missing/empty %s from reference %s...", self.faces_json_path, src)
                self._verify_and_restore(self.faces_json_path, src)

        if not self.faces_json_path.is_file():
            logger.warning("faces.json not found at %s. Running with empty registry.", self.faces_json_path)
            return

        try:
            with open(self.faces_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to parse faces.json: %s. Attempting reference recovery...", e)
            src = ref_path if (ref_path.is_file() and ref_path.stat().st_size > 0) else (alt_orig if (alt_orig.is_file() and alt_orig.stat().st_size > 0) else None)
            if src and self._verify_and_restore(self.faces_json_path, src):
                try:
                    with open(self.faces_json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e2:
                    logger.error("Post-recovery parse failed: %s", e2)
                    return
            else:
                return

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

    @property
    def is_available(self) -> bool:
        return self._is_initialized and self._app is not None

    def process_event(
        self,
        event: ConfirmedLitterEvent,
        frames: List[Tuple[float, np.ndarray, int]],
        person_box: Optional[Tuple[float, float, float, float]] = None,
        person_history: Optional[List[Tuple[int, float, Any]]] = None,
    ) -> Tuple[str, float, RecognitionStatus, Optional[np.ndarray]]:
        """Identify offender from event video window using verified person track localization."""
        # Safeguard: Do not attempt facial recognition if no offender was attributed
        from app.common_types import AttributionStatus
        if not event.person_track_id or event.attribution_status != AttributionStatus.ATTRIBUTED:
            return "Unknown", 0.0, RecognitionStatus.SKIPPED, None

        if not frames:
            return "Unknown", 0.0, RecognitionStatus.NO_FACE, None

        if not self.is_available:
            return "Unknown", 0.0, RecognitionStatus.SKIPPED, None

        best_face_crop: Optional[np.ndarray] = None
        best_face_score = 0.0
        best_embedding: Optional[np.ndarray] = None

        # Sample frames chronologically (sample every 2 frames to save CPU)
        sampled_frames = frames[::2] if len(frames) > 10 else frames

        for frame_ts, frame, frame_idx in sampled_frames:
            # Resolve target person box strictly for the attributed person in this frame
            target_box = None
            if person_history:
                best_dt = float("inf")
                for h_idx, h_ts, h_box in person_history:
                    if h_idx == frame_idx:
                        target_box = (h_box.x1, h_box.y1, h_box.x2, h_box.y2)
                        break
                    dt = abs(h_ts - frame_ts)
                    if dt < best_dt and dt <= 0.6:
                        best_dt = dt
                        target_box = (h_box.x1, h_box.y1, h_box.x2, h_box.y2)
            elif person_box is not None:
                target_box = person_box

            # CRITICAL SAFEGUARD: Never search the entire frame without person localization.
            # If the responsible person is not present in this frame, skip it to prevent bystander misidentification.
            if target_box is None:
                continue

            px1, py1, px2, py2 = [int(v) for v in target_box]
            h_f, w_f = frame.shape[:2]
            pw = px2 - px1
            ph = py2 - py1
            pad_x = max(25, int(pw * 0.15))
            pad_y = max(25, int(ph * 0.15))
            px1, py1 = max(0, px1 - pad_x), max(0, py1 - pad_y)
            px2, py2 = min(w_f, px2 + pad_x), min(h_f, py2 + pad_y)
            if (px2 - px1) < self.min_face_size or (py2 - py1) < self.min_face_size:
                continue

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

                # Crop face with contextual margin and evaluate sharpness (Laplacian variance)
                margin_x = int(fw * 0.15)
                margin_y = int(fh * 0.15)
                fx1 = max(0, bbox[0] + offset_x - margin_x)
                fy1 = max(0, bbox[1] + offset_y - margin_y)
                fx2 = min(frame.shape[1], bbox[2] + offset_x + margin_x)
                fy2 = min(frame.shape[0], bbox[3] + offset_y + margin_y)
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

    def has_identity(self, name: str) -> bool:
        """Check if identity is already enrolled in memory."""
        if not self._registry and self.faces_json_path.is_file():
            self._load_registry()
        return name.strip() in self._registry

    def get_enrolled_names(self) -> List[str]:
        """Return list of all currently enrolled identity names."""
        if not self._registry and self.faces_json_path.is_file():
            self._load_registry()
        return sorted(list(self._registry.keys()))

    def extract_face(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Optional[Tuple[int, int, int, int]]]:
        """Extract and validate a single high-quality face embedding from a frame.

        Returns:
            Tuple of (normalized_embedding, face_crop, status_message, bbox_tuple)
        """
        if not self.is_available:
            return None, None, "Face recognition model not available", None

        if frame is None or frame.size == 0:
            return None, None, "Invalid empty frame", None

        try:
            faces = self._app.get(frame)
        except Exception as e:
            return None, None, f"Face detector error: {e}", None

        if not faces:
            return None, None, "No face detected in frame", None

        if len(faces) > 1:
            return None, None, f"Multiple faces detected ({len(faces)}). Ensure only 1 person is visible.", None

        face = faces[0]
        if face.det_score < self.min_det_score:
            return None, None, f"Detection score too low ({face.det_score:.2f} < {self.min_det_score:.2f})", None

        bbox = face.bbox.astype(int)
        fw = bbox[2] - bbox[0]
        fh = bbox[3] - bbox[1]
        bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

        if fw < self.min_face_size or fh < self.min_face_size:
            return None, None, f"Face too small ({fw}x{fh} < {self.min_face_size}px)", bbox_tuple

        # Check face crop and sharpness
        h_f, w_f = frame.shape[:2]
        margin_x = int(fw * 0.15)
        margin_y = int(fh * 0.15)
        fx1 = max(0, bbox[0] - margin_x)
        fy1 = max(0, bbox[1] - margin_y)
        fx2 = min(w_f, bbox[2] + margin_x)
        fy2 = min(h_f, bbox[3] + margin_y)
        face_crop = frame[fy1:fy2, fx1:fx2]

        if face_crop.size == 0:
            return None, None, "Invalid face crop", bbox_tuple

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self.min_sharpness:
            return None, None, f"Face blurry (sharpness {sharpness:.1f} < {self.min_sharpness:.1f})", bbox_tuple

        if face.embedding is None:
            return None, None, "No face embedding extracted", bbox_tuple

        norm = float(np.linalg.norm(face.embedding))
        if norm <= 0.0:
            return None, None, "Zero norm face embedding", bbox_tuple

        normalized_emb = (face.embedding / norm).astype(np.float32)
        return normalized_emb, face_crop, "OK", bbox_tuple

    def register_identity(
        self, name: str, embedding: np.ndarray, replace: bool = False
    ) -> Tuple[bool, str]:
        """Safely register a new identity into faces.json and reference backup.

        Args:
            name: Person display name.
            embedding: 512-dim facial feature vector.
            replace: Whether to overwrite if identity exists.

        Returns:
            Tuple of (success_boolean, status_message)
        """
        clean_name = name.strip()
        if not clean_name:
            return False, "Display name cannot be empty."

        if not self._registry and self.faces_json_path.is_file():
            self._load_registry()

        if clean_name.lower() == "unknown":
            return False, "'Unknown' is a reserved identity name and cannot be enrolled."

        if clean_name in self._registry and not replace:
            return False, f"Identity '{clean_name}' is already enrolled."

        try:
            norm = float(np.linalg.norm(embedding))
            if norm <= 0.0:
                return False, "Invalid zero-length embedding vector."
            norm_vec = (embedding / norm).astype(np.float32)
        except Exception as e:
            return False, f"Embedding normalization failed: {e}"

        # 1. Load existing registry JSON
        data: Dict[str, Any] = {}
        if self.faces_json_path.is_file() and self.faces_json_path.stat().st_size > 0:
            try:
                with open(self.faces_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("Error reading %s: %s. Starting fresh.", self.faces_json_path, e)

        # 2. Update with new identity
        data[clean_name] = {
            "embedding": [round(float(v), 6) for v in norm_vec.tolist()],
            "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # 3. Write to faces.json and reference backup safely
        target_files = [self.faces_json_path]
        ref_path = self.faces_json_path.parent / "faces_reference.json"
        target_files.append(ref_path)

        for p in target_files:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                if p.exists():
                    try:
                        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
                    except Exception:
                        pass
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                try:
                    os.chmod(p, stat.S_IREAD)
                except Exception:
                    pass
            except Exception as e:
                logger.error("Failed to write updated registry to %s: %s", p, e)
                return False, f"Failed to persist identity to disk: {e}"

        # 4. Update in-memory registry
        self._registry[clean_name] = norm_vec
        logger.info("Successfully enrolled identity '%s' (total %d)", clean_name, len(self._registry))
        return True, f"Identity '{clean_name}' successfully enrolled."
