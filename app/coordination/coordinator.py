"""Application Coordinator orchestrating camera, detectors, tracking, association, recognition, and storage."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Callable, List, Optional
import cv2
import numpy as np
import yaml

from app.association.association_engine import PersonWasteAssociationEngine
from app.association.zones import ZoneManager
from app.capture.camera_service import SharedCameraService
from app.common_types import ConfirmedLitterEvent, TrackedObject
from app.detection.person_detector import PersonDetector
from app.detection.waste_detector import WasteDetector
from app.evidence.recorder import EvidenceRecorder
from app.recognition.face_adapter import FaceRecognitionAdapter
from app.storage.csv_storage import CSVEventStorage
from app.tracking.tracker import MultiObjectTrackingService

logger = logging.getLogger(__name__)


class ApplicationCoordinator:
    """Coordinates the integrated Waste Detection and Face Recognition pipeline."""

    def __init__(
        self,
        settings_path: str | Path = "config/settings.yaml",
        zones_path: str | Path = "config/zones.yaml",
    ) -> None:
        self.settings_path = Path(settings_path)
        self.zones_path = Path(zones_path)
        self.config: dict = {}
        self._load_settings()

        # Core services
        self.camera: Optional[SharedCameraService] = None
        self.waste_detector: Optional[WasteDetector] = None
        self.person_detector: Optional[PersonDetector] = None
        self.tracker: Optional[MultiObjectTrackingService] = None
        self.zones: Optional[ZoneManager] = None
        self.association_engine: Optional[PersonWasteAssociationEngine] = None
        self.face_adapter: Optional[FaceRecognitionAdapter] = None
        self.evidence_recorder: Optional[EvidenceRecorder] = None
        self.storage: Optional[CSVEventStorage] = None

        self.is_running = False
        self.frame_count = 0
        self.total_events = 0
        self.processing_fps = 0.0

    def _load_settings(self) -> None:
        """Load YAML configuration."""
        if not self.settings_path.is_file():
            raise FileNotFoundError(f"Configuration not found at {self.settings_path}")
        with open(self.settings_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def initialize(self, source_override: Optional[str] = None) -> bool:
        """Instantiate and initialize all pipeline components."""
        logger.info("Initializing Waste_dumping_v2 System...")
        cam_cfg = self.config.get("camera", {})
        models_cfg = self.config.get("models", {})
        track_cfg = self.config.get("tracking", {})
        assoc_cfg = self.config.get("association", {})
        rec_cfg = self.config.get("recognition", {})
        evi_cfg = self.config.get("evidence", {})
        stor_cfg = self.config.get("storage", {})

        # 1. Camera service
        source = source_override or cam_cfg.get("default_source", "0")
        self.camera = SharedCameraService(
            source=source,
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30.0),
            buffer_duration_seconds=cam_cfg.get("buffer_duration_seconds", 12.0),
        )

        # 2. Detectors
        w_cfg = models_cfg.get("waste", {})
        self.waste_detector = WasteDetector(
            model_path=w_cfg.get("model_path", "models/waste/waste_detection_yolo11m.pt"),
            confidence_threshold=w_cfg.get("confidence_threshold", 0.25),
            image_size=w_cfg.get("image_size", 416),
            device=self.config.get("system", {}).get("device", "cpu"),
            verify_sha256=True,
        )

        p_cfg = models_cfg.get("person", {})
        self.person_detector = PersonDetector(
            model_path=p_cfg.get("model_path", "models/person/person_detector.pt"),
            confidence_threshold=p_cfg.get("confidence_threshold", 0.35),
            image_size=p_cfg.get("image_size", 416),
            device=self.config.get("system", {}).get("device", "cpu"),
            person_class_id=p_cfg.get("person_class_id", 0),
        )

        # 3. Tracker
        self.tracker = MultiObjectTrackingService(
            high_thresh=track_cfg.get("track_high_thresh", 0.30),
            low_thresh=track_cfg.get("track_low_thresh", 0.10),
            match_thresh=track_cfg.get("match_thresh", 0.70),
            track_buffer=track_cfg.get("track_buffer", 30),
        )

        # 4. Zone Manager & Association Engine
        self.zones = ZoneManager(self.zones_path)
        self.association_engine = PersonWasteAssociationEngine(
            zone_manager=self.zones,
            holding_max_distance_px=assoc_cfg.get("holding_max_distance_px", 160.0),
            holding_iou_threshold=assoc_cfg.get("holding_iou_threshold", 0.05),
            min_holding_frames=assoc_cfg.get("min_holding_frames", 2),
            separation_distance_delta_px=assoc_cfg.get("separation_distance_delta_px", 20.0),
            stationary_duration_seconds=assoc_cfg.get("stationary_duration_seconds", 3.0),
            event_cooldown_seconds=assoc_cfg.get("event_cooldown_seconds", 4.0),
            ambiguity_margin_px=assoc_cfg.get("ambiguity_margin_px", 35.0),
        )

        # 5. Face Recognition Adapter
        self.face_adapter = FaceRecognitionAdapter(
            faces_json_path=rec_cfg.get("faces_json_path", "data/faces.json"),
            model_name=rec_cfg.get("model_name", "buffalo_s"),
            similarity_threshold=rec_cfg.get("similarity_threshold", 0.50),
            ambiguity_margin=rec_cfg.get("ambiguity_margin", 0.03),
            min_det_score=rec_cfg.get("min_det_score", 0.35),
            min_face_size_px=rec_cfg.get("min_face_size_px", 30),
            min_sharpness=rec_cfg.get("min_sharpness", 12.0),
        )
        self.face_adapter.initialize()

        # 6. Evidence Recorder & CSV Storage
        self.evidence_recorder = EvidenceRecorder(
            evidence_dir=evi_cfg.get("output_dir", "data/evidence"),
            frame_buffer=self.camera.buffer,
            pre_event_seconds=evi_cfg.get("pre_event_seconds", 5.0),
            post_event_seconds=evi_cfg.get("post_event_seconds", 5.0),
            codec=evi_cfg.get("video_codec", "mp4v"),
        )
        self.storage = CSVEventStorage(stor_cfg.get("csv_path", "data/events.csv"))

        logger.info("All pipeline components initialized successfully.")
        return True

    def run(
        self,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[np.ndarray, List[TrackedObject], List[TrackedObject], List[ConfirmedLitterEvent]], None]] = None,
    ) -> None:
        """Execute processing loop over video stream."""
        if not self.camera or not self.camera.open():
            logger.error("Camera could not be opened. Aborting run.")
            return

        self.is_running = True
        self.frame_count = 0
        t_start = time.perf_counter()

        logger.info("Starting processing loop...")
        try:
            for frame, ts, f_idx in self.camera.stream():
                t_frame_start = time.perf_counter()
                self.frame_count += 1

                # 1. Person Detection
                person_dets = self.person_detector.detect(frame)
                person_boxes = [d.bbox for d in person_dets]

                # 2. Waste Detection with Non-Waste Rejection Filter
                waste_dets = self.waste_detector.detect(frame, person_boxes=person_boxes)

                # 3. Multi-Object Tracking
                tracked_persons, tracked_waste = self.tracker.update(person_dets, waste_dets, f_idx, ts)

                # 4. Spatiotemporal Association
                h_f, w_f = frame.shape[:2]
                new_events = self.association_engine.process_frame(
                    persons=tracked_persons,
                    waste_items=tracked_waste,
                    frame_w=w_f,
                    frame_h=h_f,
                    frame_idx=f_idx,
                    timestamp=ts,
                )

                # 5. For each confirmed event: Trigger Face Recognition & Schedule Evidence
                for event in new_events:
                    self.total_events += 1
                    t_rec_start = time.perf_counter()
                    # Retrieve associated person bounding box if available
                    person_box_coords = None
                    if event.person_track_id:
                        p_obj = next((p for p in tracked_persons if p.track_id == event.person_track_id), None)
                        if p_obj:
                            person_box_coords = (p_obj.bbox.x1, p_obj.bbox.y1, p_obj.bbox.x2, p_obj.bbox.y2)

                    # Get recent window around event
                    recent_frames = self.camera.buffer.get_window(ts - 2.5, ts + 0.5)
                    id_name, sim, r_stat, face_crop = self.face_adapter.process_event(
                        event=event,
                        frames=recent_frames,
                        person_box=person_box_coords,
                    )
                    event.recognized_person = id_name
                    event.recognition_status = r_stat
                    event.processing_latency_ms = (time.perf_counter() - t_rec_start) * 1000

                    # Schedule video clip and snapshot save
                    self.evidence_recorder.schedule_clip(event, face_crop)

                # 6. Check pending evidence clips whose post-event window has elapsed
                finalized_events = self.evidence_recorder.check_pending(ts)
                for fin_evt in finalized_events:
                    self.storage.append_event(fin_evt)

                # Latency & FPS measurement
                dt = time.perf_counter() - t_frame_start
                self.processing_fps = 1.0 / dt if dt > 0 else 0.0

                if callback:
                    callback(frame, tracked_persons, tracked_waste, new_events)

                if max_frames and self.frame_count >= max_frames:
                    logger.info("Reached requested max_frames limit (%d)", max_frames)
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.stop()
            t_total = time.perf_counter() - t_start
            avg_fps = self.frame_count / t_total if t_total > 0 else 0.0
            logger.info("Processing stopped. Processed %d frames in %.2fs (Average FPS: %.2f)", self.frame_count, t_total, avg_fps)

    def stop(self) -> None:
        """Flush evidence, append remaining events, and release camera."""
        self.is_running = False
        if self.evidence_recorder and self.storage:
            flushed = self.evidence_recorder.flush_all()
            for evt in flushed:
                self.storage.append_event(evt)

        if self.camera:
            self.camera.close()
