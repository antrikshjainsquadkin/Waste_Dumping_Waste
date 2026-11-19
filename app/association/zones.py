"""Zone Management for Open-Ground and Dustbin Polygons."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml

from app.common_types import BoundingBox

logger = logging.getLogger(__name__)


class DetectionZone:
    """Represents a geometric polygon zone (open-ground or bin receptacle)."""

    def __init__(
        self,
        zone_id: str,
        zone_type: str,  # 'ground' or 'bin'
        polygon: List[List[float]],  # [[x1, y1], [x2, y2], ...]
        is_normalized: bool = True,
        description: str = "",
    ) -> None:
        self.zone_id = zone_id
        self.zone_type = zone_type.lower()
        self.polygon_coords = polygon
        self.is_normalized = is_normalized
        self.description = description

    def contains_point(self, pt: Tuple[float, float], frame_w: int, frame_h: int) -> bool:
        """Check if point (x, y) falls inside the polygon."""
        if not self.polygon_coords or len(self.polygon_coords) < 3:
            return False

        poly_pts = []
        for x, y in self.polygon_coords:
            if self.is_normalized:
                poly_pts.append([x * frame_w, y * frame_h])
            else:
                poly_pts.append([x, y])

        poly_np = np.array(poly_pts, dtype=np.int32)
        dist = cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), measureDist=False)
        return dist >= 0

    def contains_box(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> bool:
        """Check if bottom center of bounding box touches zone."""
        # For ground contact, the bottom edge / bottom-center of the object touches the ground
        bottom_center = ((bbox.x1 + bbox.x2) / 2.0, bbox.y2)
        return self.contains_point(bottom_center, frame_w, frame_h)


class ZoneManager:
    """Loads and manages configurable zones from YAML file."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.zones: Dict[str, DetectionZone] = {}
        self.load_zones()

    def load_zones(self) -> None:
        """Load zone definitions from YAML."""
        if not self.config_path.is_file():
            logger.warning("Zone config %s not found. Using default open-ground.", self.config_path)
            self._create_default_zones()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            raw_zones = data.get("zones", {})
            for zid, zdata in raw_zones.items():
                zone = DetectionZone(
                    zone_id=zdata.get("zone_id", zid),
                    zone_type=zdata.get("zone_type", "ground"),
                    polygon=zdata.get("polygon", []),
                    is_normalized=zdata.get("is_normalized", True),
                    description=zdata.get("description", ""),
                )
                self.zones[zone.zone_id] = zone
            logger.info("Loaded %d zones from %s", len(self.zones), self.config_path)
        except Exception as e:
            logger.error("Error loading zones from %s: %s", self.config_path, e)
            self._create_default_zones()

    def _create_default_zones(self) -> None:
        """Fallback default ground zone covering bottom 75% of view."""
        self.zones["open_ground_default"] = DetectionZone(
            zone_id="open_ground_default",
            zone_type="ground",
            polygon=[[0.0, 0.25], [1.0, 0.25], [1.0, 1.0], [0.0, 1.0]],
            is_normalized=True,
            description="Default floor/ground polygon",
        )

    def find_matching_bin(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> Optional[DetectionZone]:
        """Check if bbox enters any bin zone."""
        for zone in self.zones.values():
            if zone.zone_type == "bin" and zone.contains_box(bbox, frame_w, frame_h):
                return zone
        return None

    def is_in_ground_zone(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> Tuple[bool, Optional[str]]:
        """Check if bbox is on a configured ground zone."""
        for zone in self.zones.values():
            if zone.zone_type == "ground" and zone.contains_box(bbox, frame_w, frame_h):
                return True, zone.zone_id
        return False, None
