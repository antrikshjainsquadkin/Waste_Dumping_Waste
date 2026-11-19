"""Core domain types, enums, and dataclasses for Waste_dumping_v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class DisposalEventType(str, Enum):
    OPEN_GROUND_LITTERING = "OPEN_GROUND_LITTERING"
    BIN_DISPOSAL = "BIN_DISPOSAL"


class LitteringState(str, Enum):
    WASTE_DETECTED = "WASTE_DETECTED"
    PERSON_INTERACTION_CANDIDATE = "PERSON_INTERACTION_CANDIDATE"
    PERSON_ASSOCIATED = "PERSON_ASSOCIATED"
    RELEASE_SUSPECTED = "RELEASE_SUSPECTED"
    GROUND_CONTACT = "GROUND_CONTACT"
    STATIONARY_CONFIRMATION = "STATIONARY_CONFIRMATION"
    LITTERING_EVENT_CONFIRMED = "LITTERING_EVENT_CONFIRMED"
    EVIDENCE_CAPTURED = "EVIDENCE_CAPTURED"
    EVENT_SAVED = "EVENT_SAVED"


class AttributionStatus(str, Enum):
    ATTRIBUTED = "ATTRIBUTED"
    UNATTRIBUTED = "UNATTRIBUTED"
    AMBIGUOUS = "AMBIGUOUS"


class RecognitionStatus(str, Enum):
    IDENTIFIED = "IDENTIFIED"
    UNKNOWN = "UNKNOWN"
    NO_FACE = "NO_FACE"
    SKIPPED = "SKIPPED"


@dataclass(slots=True, frozen=True)
class BoundingBox:
    """Bounding box coordinates (x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_xyxy_int(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    def to_xywh(self) -> Tuple[float, float, float, float]:
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def iou(self, other: BoundingBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def ioa(self, other: BoundingBox) -> float:
        """Intersection over Self Area: proportion of this box contained in other."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        return inter / self.area if self.area > 0 else 0.0

    def distance_to(self, other: BoundingBox) -> float:
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return float(np.hypot(cx1 - cx2, cy1 - cy2))


@dataclass(slots=True)
class Detection:
    """Raw object detection output."""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass(slots=True)
class TrackedObject:
    """Object tracked across video frames."""
    track_id: str  # e.g., 'Person_1' or 'Waste_2'
    raw_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    last_frame_idx: int
    last_timestamp: float
    history: List[Tuple[int, float, BoundingBox]] = field(default_factory=list)

    def update(self, bbox: BoundingBox, conf: float, frame_idx: int, timestamp: float) -> None:
        self.bbox = bbox
        self.confidence = conf
        self.last_frame_idx = frame_idx
        self.last_timestamp = timestamp
        self.history.append((frame_idx, timestamp, bbox))
        if len(self.history) > 150:
            self.history = self.history[-150:]


@dataclass
class ConfirmedLitterEvent:
    """Event payload emitted when disposal/littering is confirmed."""
    event_id: str
    event_type: DisposalEventType
    estimated_release_time: str
    estimated_release_timestamp: float
    confirmed_at: str
    confirmed_timestamp: float
    waste_category: str
    waste_confidence: float
    waste_track_id: str
    person_track_id: Optional[str]
    recognized_person: str = "Unknown"
    attribution_status: AttributionStatus = AttributionStatus.UNATTRIBUTED
    recognition_status: RecognitionStatus = RecognitionStatus.UNKNOWN
    ground_zone_id: str = "open_ground_default"
    video_evidence_path: str = ""
    face_snapshot_path: str = ""
    plate_snapshot_path: str = ""
    processing_latency_ms: float = 0.0
    notes: str = ""
