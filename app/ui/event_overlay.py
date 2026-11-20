"""Dynamic Event-Status Overlay for Waste Dumping Detection V2.

Renders an on-screen heads-up display (HUD) panel on detection video frames
matching the operator visual specification:
- Dark semi-transparent background.
- Orange border.
- Orange bold uppercase event heading (DUMPING ACTIVITY DETECTED).
- Light-colored metadata text (Person Track, Waste, Status, Identity).
- Highlighted orange review-status text (REVIEW REQUIRED).
- Distinct non-alert representation for BIN_DISPOSAL.
- Compact neutral monitoring state when no dumping event is active.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.common_types import (
    AttributionStatus,
    ConfirmedLitterEvent,
    DisposalEventType,
    RecognitionStatus,
    TrackedObject,
)


class EventStatusOverlay:
    """Renderer and state manager for dynamic on-frame event status HUD."""

    # Color Palette in OpenCV BGR
    COLOR_BG = (22, 22, 26)              # Dark slate / charcoal background
    COLOR_BORDER_ALERT = (0, 140, 255)   # Vivid Safety Orange
    COLOR_BORDER_BIN = (0, 200, 120)     # Compliant Green
    COLOR_BORDER_IDLE = (70, 75, 80)     # Muted Slate
    COLOR_HEADING_ALERT = (0, 155, 255)  # Bright Orange
    COLOR_HEADING_BIN = (0, 220, 130)    # Soft Green
    COLOR_HEADING_IDLE = (120, 220, 140) # Muted Green
    COLOR_LABEL = (185, 190, 195)        # Light Cool Gray for labels
    COLOR_VALUE = (245, 245, 245)        # Off-White for values
    COLOR_STATUS_ALERT = (0, 140, 255)   # Orange for REVIEW REQUIRED
    COLOR_STATUS_BIN = (0, 220, 130)     # Green for COMPLIANT / IN BIN

    def __init__(
        self,
        display_duration_seconds: float = 12.0,
        alpha: float = 0.82,
        margin_x: int = 16,
        margin_y: int = 16,
    ) -> None:
        """Initialize overlay with configurable persistence duration and layout.

        Args:
            display_duration_seconds: How long a confirmed event remains prominent
                                      before transitioning to idle monitoring.
            alpha: Opacity factor for the background rectangle (0.0 to 1.0).
            margin_x: Distance in pixels from left viewport edge.
            margin_y: Distance in pixels from top viewport edge.
        """
        self.display_duration_seconds = float(display_duration_seconds)
        self.alpha = float(np.clip(alpha, 0.1, 1.0))
        self.margin_x = int(margin_x)
        self.margin_y = int(margin_y)

        self._active_event: Optional[ConfirmedLitterEvent] = None
        self._last_event_time: float = 0.0

    @property
    def active_event(self) -> Optional[ConfirmedLitterEvent]:
        """Return the currently displayed active event if within duration."""
        if self._active_event is None:
            return None
        if (time.time() - self._last_event_time) > self.display_duration_seconds:
            return None
        return self._active_event

    def update_event(self, event: ConfirmedLitterEvent) -> None:
        """Push a newly confirmed event into the overlay state."""
        self._active_event = event
        self._last_event_time = time.time()

    def clear(self) -> None:
        """Manually clear the active event and revert to idle state."""
        self._active_event = None
        self._last_event_time = 0.0

    def draw(
        self,
        frame: np.ndarray,
        new_events: Optional[List[ConfirmedLitterEvent]] = None,
        tracked_persons: Optional[List[TrackedObject]] = None,
        tracked_waste: Optional[List[TrackedObject]] = None,
    ) -> np.ndarray:
        """Draw the status overlay directly onto the provided frame.

        Args:
            frame: BGR image array (modified in-place and returned).
            new_events: Optional list of new events emitted on this frame.
            tracked_persons: Optional list of currently tracked persons (for count).
            tracked_waste: Optional list of currently tracked waste items.

        Returns:
            The modified frame with the status overlay rendered.
        """
        if frame is None or frame.size == 0:
            return frame

        # Ingest any newly arrived confirmed event
        if new_events:
            for ev in new_events:
                if ev:
                    self.update_event(ev)

        fh, fw = frame.shape[:2]

        # Determine whether we render Active Event or Idle Monitoring
        curr_event = self.active_event
        if curr_event is not None:
            return self._draw_active_event(frame, curr_event, fw, fh)
        else:
            return self._draw_idle_monitoring(
                frame,
                fw,
                fh,
                person_count=len(tracked_persons) if tracked_persons else 0,
                waste_count=len(tracked_waste) if tracked_waste else 0,
            )

    # -------------------------------------------------------------------------
    # Active Event Rendering
    # -------------------------------------------------------------------------
    def _draw_active_event(
        self,
        frame: np.ndarray,
        event: ConfirmedLitterEvent,
        fw: int,
        fh: int,
    ) -> np.ndarray:
        """Render the full rectangular status panel for an active confirmed event."""
        is_bin_disposal = (event.event_type == DisposalEventType.BIN_DISPOSAL)

        # 1. Heading and border color
        if is_bin_disposal:
            heading = "BIN DISPOSAL DETECTED"
            border_color = self.COLOR_BORDER_BIN
            heading_color = self.COLOR_HEADING_BIN
            status_text = "COMPLIANT / IN BIN"
            status_color = self.COLOR_STATUS_BIN
        else:
            heading = "DUMPING ACTIVITY DETECTED"
            border_color = self.COLOR_BORDER_ALERT
            heading_color = self.COLOR_HEADING_ALERT
            status_text = "REVIEW REQUIRED"
            status_color = self.COLOR_STATUS_ALERT

        # 2. Format Person Track field
        if event.attribution_status == AttributionStatus.AMBIGUOUS:
            person_track_text = "Ambiguous / Multiple"
        elif event.attribution_status == AttributionStatus.UNATTRIBUTED:
            person_track_text = "Unattributed"
        elif event.person_track_id:
            person_track_text = str(event.person_track_id)
        else:
            person_track_text = "Unknown"

        # 3. Format Waste field
        waste_cat = event.waste_category.title() if event.waste_category else "Unknown"
        if event.waste_confidence > 0.0:
            waste_text = f"{waste_cat} ({event.waste_confidence:.2f})"
        else:
            waste_text = waste_cat

        # 4. Format Identity field
        if (
            event.recognition_status == RecognitionStatus.IDENTIFIED
            and event.recognized_person
            and event.recognized_person != "Unknown"
        ):
            identity_text = str(event.recognized_person)
        elif event.recognition_status == RecognitionStatus.NO_FACE:
            identity_text = "No Face Visible"
        elif event.recognition_status == RecognitionStatus.SKIPPED:
            identity_text = "Skipped"
        else:
            identity_text = "Unknown"

        # Content lines: (Label, Value, Value Color)
        lines = [
            ("Person Track: ", person_track_text, self.COLOR_VALUE),
            ("Waste: ", waste_text, self.COLOR_VALUE),
            ("Status: ", status_text, status_color),
            ("Identity: ", identity_text, self.COLOR_VALUE),
        ]

        # 5. Dynamic panel dimensions
        # Width: bounded between 320 and (fw - margin*2)
        panel_w = min(360, max(280, fw - (self.margin_x * 2)))
        panel_h = 168

        # Ensure panel fits vertically
        x1 = self.margin_x
        y1 = self.margin_y
        x2 = min(fw - 4, x1 + panel_w)
        y2 = min(fh - 4, y1 + panel_h)

        if (x2 - x1) < 100 or (y2 - y1) < 60:
            return frame  # Viewport too small for overlay

        # 6. Semi-transparent background blending
        self._render_translucent_box(frame, x1, y1, x2, y2, self.COLOR_BG, self.alpha)

        # 7. High-visibility solid border
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)

        # 8. Render Heading
        text_x = x1 + 14
        heading_y = y1 + 28
        cv2.putText(
            frame,
            heading,
            (text_x, heading_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            heading_color,
            2,
            cv2.LINE_AA,
        )

        # Thin dividing accent line under heading
        div_y = y1 + 38
        cv2.line(frame, (x1 + 10, div_y), (x2 - 10, div_y), border_color, 1, cv2.LINE_AA)

        # 9. Render key-value metadata rows
        row_y = y1 + 64
        line_spacing = 25

        for label, val, val_col in lines:
            # Draw label
            cv2.putText(
                frame,
                label,
                (text_x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                self.COLOR_LABEL,
                1,
                cv2.LINE_AA,
            )
            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)

            # Draw value (bold if alert status)
            val_weight = 2 if val_col == status_color and status_text == "REVIEW REQUIRED" else 1
            cv2.putText(
                frame,
                val,
                (text_x + lw, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                val_col,
                val_weight,
                cv2.LINE_AA,
            )
            row_y += line_spacing

        return frame

    # -------------------------------------------------------------------------
    # Compact Idle Monitoring State
    # -------------------------------------------------------------------------
    def _draw_idle_monitoring(
        self,
        frame: np.ndarray,
        fw: int,
        fh: int,
        person_count: int = 0,
        waste_count: int = 0,
    ) -> np.ndarray:
        """Render a clean, non-intrusive idle status bar when no alert is active."""
        panel_w = min(360, max(260, fw - (self.margin_x * 2)))
        panel_h = 52

        x1 = self.margin_x
        y1 = self.margin_y
        x2 = min(fw - 4, x1 + panel_w)
        y2 = min(fh - 4, y1 + panel_h)

        if (x2 - x1) < 100 or (y2 - y1) < 30:
            return frame

        # Semi-transparent background
        self._render_translucent_box(frame, x1, y1, x2, y2, self.COLOR_BG, 0.75)
        # Subtle border
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_BORDER_IDLE, 1, cv2.LINE_AA)

        text_x = x1 + 12
        # Green status indicator pip
        cv2.circle(frame, (text_x + 4, y1 + 18), 4, self.COLOR_HEADING_IDLE, -1, cv2.LINE_AA)

        # Header title
        cv2.putText(
            frame,
            "MONITORING - NO INCIDENTS",
            (text_x + 16, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            self.COLOR_HEADING_IDLE,
            1,
            cv2.LINE_AA,
        )

        # Subtitle metrics
        metrics_text = f"Active Scene: {person_count} Person(s) | {waste_count} Waste Item(s)"
        cv2.putText(
            frame,
            metrics_text,
            (text_x + 16, y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            self.COLOR_LABEL,
            1,
            cv2.LINE_AA,
        )

        return frame

    # -------------------------------------------------------------------------
    # Alpha Blending Helper
    # -------------------------------------------------------------------------
    @staticmethod
    def _render_translucent_box(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color_bgr: Tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Apply alpha-blended filled rectangle on ROI without copying entire frame."""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        box = np.full_like(roi, color_bgr, dtype=np.uint8)
        cv2.addWeighted(box, alpha, roi, 1.0 - alpha, 0, dst=roi)
