"""Command-line entry point for Waste_dumping_v2.

Provides terminal workflow for source selection, zone monitoring,
live console reporting, and evidence recording.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.coordination.coordinator import ApplicationCoordinator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("WasteDumpingV2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Waste Detection & Face Recognition V2 — Open-Ground & Bin Disposal System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index ('0', '1'), path to video file (.mp4, .avi), or RTSP URL",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "settings.yaml"),
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--zones",
        type=str,
        default=str(PROJECT_ROOT / "config" / "zones.yaml"),
        help="Path to zones.yaml",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable OpenCV GUI preview window (recommended for headless or background execution)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process before exiting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 75)
    print("      WASTE DETECTION & FACE RECOGNITION V2 ARCHITECTURE")
    print("=" * 75)
    print(f" Source:            {args.source}")
    print(f" Settings Config:   {args.config}")
    print(f" Zones Config:      {args.zones}")
    print(f" Preview Window:    {'Disabled (Headless)' if args.no_preview else 'Enabled'}")
    print("=" * 75)

    coordinator = ApplicationCoordinator(
        settings_path=args.config,
        zones_path=args.zones,
    )

    if not coordinator.initialize(source_override=args.source):
        logger.error("Failed to initialize coordinator. Exiting.")
        return 1

    last_log_time = time.time()

    def frame_callback(frame, persons, waste, events):
        nonlocal last_log_time
        # Periodic console status log every 1.5 seconds
        if time.time() - last_log_time > 1.5:
            last_log_time = time.time()
            p_ids = [p.track_id for p in persons]
            w_ids = [w.track_id for w in waste]
            print(
                f"[Frame {coordinator.frame_count:05d}] "
                f"FPS: {coordinator.processing_fps:.1f} | "
                f"People: {len(persons)} {p_ids} | "
                f"Waste: {len(waste)} {w_ids} | "
                f"Events: {coordinator.total_events} | "
                f"CSV Records: {coordinator.storage.total_records}"
            )

        if events:
            for ev in events:
                print("\n" + "!" * 70)
                print(f" [LITTERING CONFIRMED] {ev.event_type.value}")
                print(f" Event ID:          {ev.event_id}")
                print(f" Responsible Person: {ev.person_track_id} ({ev.attribution_status.value})")
                print(f" Recognized Face:    {ev.recognized_person} ({ev.recognition_status.value})")
                print(f" Waste Track:        {ev.waste_track_id} ({ev.waste_category}, conf={ev.waste_confidence:.2f})")
                print(f" Zone:               {ev.ground_zone_id}")
                print("!" * 70 + "\n")

        if not args.no_preview:
            # Draw visual boxes on preview frame
            for p in persons:
                x1, y1, x2, y2 = p.bbox.to_xyxy_int()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, p.track_id, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            for w in waste:
                x1, y1, x2, y2 = w.bbox.to_xyxy_int()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{w.track_id} ({w.class_name})", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("Waste_dumping_v2 Live Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                coordinator.stop()

    try:
        coordinator.run(max_frames=args.max_frames, callback=frame_callback)
    except Exception as e:
        logger.exception("Runtime error in coordinator: %s", e)
        return 1
    finally:
        if not args.no_preview:
            cv2.destroyAllWindows()

    print("\n" + "=" * 75)
    print(f" Session Complete.")
    print(f" Total Frames Processed: {coordinator.frame_count}")
    print(f" Confirmed Events:       {coordinator.total_events}")
    print(f" Records Saved in CSV:   {coordinator.storage.total_records}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
