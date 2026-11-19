"""Evaluation utility running real-world or test video through the V2 pipeline in isolation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.coordination.coordinator import ApplicationCoordinator
from app.evidence.recorder import EvidenceRecorder
from app.storage.csv_storage import CSVEventStorage

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Evaluator]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Evaluator")


def run_evaluation(
    video_path: str | Path,
    ground_truth_path: Optional[str | Path] = None,
    output_dir: str | Path = PROJECT_ROOT / "reports" / "evaluation",
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    eval_evidence_dir = out_path / "evidence"
    eval_evidence_dir.mkdir(parents=True, exist_ok=True)
    eval_csv_path = out_path / "events_eval.csv"

    # Inspect video metadata
    cap = cv2.VideoCapture(str(video_path))
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    v_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    v_duration_sec = v_total_frames / max(1.0, v_fps)
    cap.release()

    process = psutil.Process(os.getpid())
    mem_initial_mb = process.memory_info().rss / (1024 * 1024)
    peak_mem_mb = mem_initial_mb

    coordinator = ApplicationCoordinator(
        settings_path=PROJECT_ROOT / "config" / "settings.yaml",
        zones_path=PROJECT_ROOT / "config" / "zones.yaml",
    )
    coordinator.initialize(source_override=str(video_path))

    # Isolate storage and evidence recorder to avoid polluting production data
    coordinator.storage = CSVEventStorage(eval_csv_path)
    coordinator.evidence_recorder = EvidenceRecorder(
        evidence_dir=eval_evidence_dir,
        frame_buffer=coordinator.camera.buffer,
        pre_event_seconds=5.0,
        post_event_seconds=5.0,
    )

    frame_latencies: List[float] = []
    events_captured: List[Dict[str, Any]] = []

    def on_frame(frame, persons, waste, events):
        nonlocal peak_mem_mb
        current_mem = process.memory_info().rss / (1024 * 1024)
        if current_mem > peak_mem_mb:
            peak_mem_mb = current_mem

        if events:
            for ev in events:
                events_captured.append({
                    "event_id": ev.event_id,
                    "event_type": ev.event_type.value,
                    "estimated_release_time": ev.estimated_release_time,
                    "confirmed_at": ev.confirmed_at,
                    "waste_category": ev.waste_category,
                    "waste_confidence": float(ev.waste_confidence),
                    "waste_track_id": ev.waste_track_id,
                    "person_track_id": ev.person_track_id,
                    "recognized_person": ev.recognized_person,
                    "attribution_status": ev.attribution_status.value,
                    "recognition_status": ev.recognition_status.value,
                    "ground_zone_id": ev.ground_zone_id,
                })

    logger.info("Starting evaluation on %s (%d frames, %.1f FPS)...", video_path.name, v_total_frames, v_fps)
    t_start = time.perf_counter()
    coordinator.run(max_frames=max_frames, callback=on_frame)
    t_total = time.perf_counter() - t_start

    frames_processed = coordinator.frame_count
    eval_fps = frames_processed / t_total if t_total > 0 else 0.0
    mem_final_mb = process.memory_info().rss / (1024 * 1024)

    # Ground truth evaluation
    gt_comparison: Dict[str, Any] = {
        "status": "NO_GROUND_TRUTH_PROVIDED",
    }
    if ground_truth_path and Path(ground_truth_path).is_file():
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        gt_events = gt_data.get("expected_events", [])
        tp = 0
        matched_gt = set()

        for ev in events_captured:
            # Check match against any GT event within category
            for idx, gt in enumerate(gt_events):
                if idx in matched_gt:
                    continue
                if gt.get("event_type") == ev["event_type"]:
                    matched_gt.add(idx)
                    tp += 1
                    break

        fp = len(events_captured) - tp
        fn = len(gt_events) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        gt_comparison = {
            "status": "EVALUATED",
            "ground_truth_events_count": len(gt_events),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    summary = {
        "video": {
            "name": video_path.name,
            "path": str(video_path),
            "resolution": f"{v_width}x{v_height}",
            "source_fps": v_fps,
            "total_frames": v_total_frames,
            "duration_seconds": round(v_duration_sec, 2),
        },
        "performance": {
            "frames_processed": frames_processed,
            "total_processing_seconds": round(t_total, 2),
            "throughput_fps": round(eval_fps, 2),
            "memory": {
                "initial_rss_mb": round(mem_initial_mb, 2),
                "peak_rss_mb": round(peak_mem_mb, 2),
                "final_rss_mb": round(mem_final_mb, 2),
                "delta_mb": round(mem_final_mb - mem_initial_mb, 2),
            },
        },
        "events": {
            "total_confirmed": len(events_captured),
            "records": events_captured,
        },
        "ground_truth_metrics": gt_comparison,
        "evidence_files": {
            "csv_report": str(eval_csv_path),
            "clips_saved": len(list(eval_evidence_dir.glob("*.mp4"))),
            "snapshots_saved": len(list(eval_evidence_dir.glob("*.jpg"))),
        },
    }

    # Save JSON summary
    summary_path = out_path / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown report
    md_path = out_path / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Video Evaluation Report: {video_path.name}\n\n")
        f.write(f"- **Video Resolution**: {v_width}x{v_height}\n")
        f.write(f"- **Source Duration**: {v_duration_sec:.2f}s ({v_total_frames} frames @ {v_fps:.1f} fps)\n")
        f.write(f"- **Frames Evaluated**: {frames_processed}\n")
        f.write(f"- **Processing Time**: {t_total:.2f}s (Average FPS: {eval_fps:.2f})\n")
        f.write(f"- **Initial RAM**: {mem_initial_mb:.2f} MB | **Peak RAM**: {peak_mem_mb:.2f} MB\n")
        f.write(f"- **Confirmed Events**: {len(events_captured)}\n\n")

        if gt_comparison.get("status") == "EVALUATED":
            f.write("## Ground Truth Metrics\n\n")
            f.write(f"- **True Positives (TP)**: {gt_comparison['true_positives']}\n")
            f.write(f"- **False Positives (FP)**: {gt_comparison['false_positives']}\n")
            f.write(f"- **False Negatives (FN)**: {gt_comparison['false_negatives']}\n")
            f.write(f"- **Precision**: {gt_comparison['precision']:.4f}\n")
            f.write(f"- **Recall**: {gt_comparison['recall']:.4f}\n")
            f.write(f"- **F1-Score**: {gt_comparison['f1_score']:.4f}\n\n")

        f.write("## Confirmed Events Detail\n\n")
        if events_captured:
            f.write("| Event ID | Type | Release Time | Category | Conf | Person ID | Face | Attribution | Recognition |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for e in events_captured:
                f.write(f"| {e['event_id']} | {e['event_type']} | {e['estimated_release_time']} | {e['waste_category']} | {e['waste_confidence']:.2f} | {e['person_track_id']} | {e['recognized_person']} | {e['attribution_status']} | {e['recognition_status']} |\n")
        else:
            f.write("No littering or disposal events confirmed.\n")

    logger.info("Evaluation complete. Results saved to %s", out_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate video with V2 Waste Detection & Face Recognition")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--ground-truth", default=None, help="Optional ground truth JSON file")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "evaluation"), help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    res = run_evaluation(
        video_path=args.video,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
