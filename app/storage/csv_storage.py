"""Local CSV Event Storage with schema validation and duplicate prevention."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
import threading
from typing import List, Set

from app.common_types import ConfirmedLitterEvent

logger = logging.getLogger(__name__)

CSV_HEADERS: List[str] = [
    "event_id",
    "event_type",
    "estimated_release_time",
    "confirmed_at",
    "waste_category",
    "waste_confidence",
    "waste_track_id",
    "person_track_id",
    "recognized_person",
    "attribution_status",
    "recognition_status",
    "ground_zone_id",
    "video_evidence_path",
    "face_snapshot_path",
    "plate_snapshot_path",
    "processing_latency_ms",
    "notes",
]


class CSVEventStorage:
    """Thread-safe append-only CSV storage for confirmed littering and disposal events."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self._lock = threading.Lock()
        self._known_ids: Set[str] = set()
        self._init_csv()

    def _init_csv(self) -> None:
        """Create CSV file with headers if missing, or load existing event IDs."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if not self.csv_path.is_file() or self.csv_path.stat().st_size == 0:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADERS)
                logger.info("Initialized new events CSV at %s", self.csv_path)
            else:
                try:
                    with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            eid = row.get("event_id")
                            if eid:
                                self._known_ids.add(eid)
                    logger.info("Loaded %d existing event records from %s", len(self._known_ids), self.csv_path)
                except Exception as e:
                    logger.warning("Error reading existing CSV: %s", e)

    def append_event(self, event: ConfirmedLitterEvent) -> bool:
        """Append confirmed event to CSV if not duplicate."""
        with self._lock:
            if event.event_id in self._known_ids:
                logger.warning("Duplicate event_id %s rejected from CSV write.", event.event_id)
                return False

            row = [
                event.event_id,
                event.event_type.value,
                event.estimated_release_time,
                event.confirmed_at,
                event.waste_category,
                f"{event.waste_confidence:.4f}",
                event.waste_track_id,
                event.person_track_id or "None",
                event.recognized_person,
                event.attribution_status.value,
                event.recognition_status.value,
                event.ground_zone_id,
                event.video_evidence_path,
                event.face_snapshot_path,
                event.plate_snapshot_path,
                f"{event.processing_latency_ms:.2f}",
                event.notes,
            ]

            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            self._known_ids.add(event.event_id)
            logger.info("Appended event %s to CSV (%d total records)", event.event_id, len(self._known_ids))
            return True

    @property
    def total_records(self) -> int:
        with self._lock:
            return len(self._known_ids)

    def get_all_records(self) -> List[dict]:
        """Read all existing event records without modifying storage."""
        with self._lock:
            if not self.csv_path.is_file() or self.csv_path.stat().st_size == 0:
                return []
            try:
                with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            except Exception as e:
                logger.error("Failed to read CSV records from %s: %s", self.csv_path, e)
                return []
