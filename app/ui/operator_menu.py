"""Operator Terminal Menu & Input Workflow for Waste_dumping_v2.

Provides an interactive terminal interface for operators:
1. Register a person's face (via webcam enrollment).
2. Start live webcam detection.
3. Connect to an authorized CCTV camera stream (RTSP/HTTP).
4. Select and process a local video file (native Windows file picker).
5. View previously recorded event records (from CSV log).
6. Exit application safely.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.coordination.coordinator import ApplicationCoordinator
from app.recognition.face_adapter import FaceRecognitionAdapter
from app.storage.csv_storage import CSVEventStorage

logger = logging.getLogger("OperatorMenu")

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def sanitize_stream_url(url: str) -> str:
    """Mask credentials in RTSP/HTTP URLs for safe display and logging."""
    if not url:
        return ""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


def open_native_file_picker() -> Optional[str]:
    """Open native Windows file selection dialog via tkinter.filedialog.

    Returns the absolute path of the chosen file, or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.focus_force()
        file_path = filedialog.askopenfilename(
            title="Select Video File for Waste & Face Detection",
            filetypes=[
                ("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.m4v"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return str(file_path).strip() if file_path else None
    except Exception as e:
        logger.warning("Native file dialog unavailable: %s", e)
        return None


def probe_available_webcams(max_probe: int = 3) -> List[int]:
    """Probe system for available webcam device indices."""
    available = []
    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


class OperatorMenu:
    """Terminal-based operator menu orchestrating input sources and workflows."""

    def __init__(
        self,
        settings_path: str | Path = "config/settings.yaml",
        zones_path: str | Path = "config/zones.yaml",
        faces_path: str | Path = "data/faces.json",
        csv_path: str | Path = "data/events.csv",
        input_func: Callable[[str], str] = input,
        file_picker_func: Callable[[], Optional[str]] = open_native_file_picker,
    ) -> None:
        self.settings_path = Path(settings_path)
        self.zones_path = Path(zones_path)
        self.faces_path = Path(faces_path)
        self.csv_path = Path(csv_path)
        self._input = input_func
        self._file_picker = file_picker_func

    def display_menu(self) -> None:
        """Render the 6-option operator menu header."""
        print("\n" + "=" * 72)
        print("     WASTE DUMPING DETECTION & FACE RECOGNITION V2 — OPERATOR MENU")
        print("=" * 72)
        print("  [1] Register Face (Enroll new identity via webcam)")
        print("  [2] Start Live Webcam Detection")
        print("  [3] Connect to Authorized CCTV Camera (RTSP / HTTP)")
        print("  [4] Select & Process Video File (Windows File Picker)")
        print("  [5] View Event Records (CSV Log)")
        print("  [6] Exit Application")
        print("=" * 72)

    def handle_choice(self, choice: str) -> bool:
        """Route operator choice to the appropriate workflow.

        Returns:
            True to keep menu running, False to exit.
        """
        choice = choice.strip()
        if choice == "1":
            self.register_face_workflow()
            return True
        elif choice == "2":
            self.webcam_detection_workflow()
            return True
        elif choice == "3":
            self.cctv_detection_workflow()
            return True
        elif choice == "4":
            self.video_file_workflow()
            return True
        elif choice == "5":
            self.view_records_workflow()
            return True
        elif choice == "6":
            return self.exit_workflow()
        else:
            print(f"\n[ERROR] Invalid choice '{choice}'. Please select a number between 1 and 6.")
            return True

    def run_menu_loop(self) -> None:
        """Interactive loop running the operator menu."""
        while True:
            self.display_menu()
            try:
                choice = self._input("Select an option [1-6]: ")
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] Exiting Operator Menu.")
                break

            if not self.handle_choice(choice):
                break

    # -------------------------------------------------------------------------
    # Option 1: Register Face
    # -------------------------------------------------------------------------
    def register_face_workflow(self, camera_index: int = 0) -> bool:
        """Capture and enroll a new person's face using webcam and InsightFace."""
        print("\n" + "=" * 65)
        print("           OPTION 1: FACE REGISTRATION & ENROLLMENT")
        print("=" * 65)
        try:
            name = self._input("Enter person's display name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Registration cancelled.")
            return False

        if not name:
            print("[ERROR] Display name cannot be empty. Returning to menu.\n")
            return False
        if name.lower() == "unknown":
            print("[ERROR] 'Unknown' is a reserved identity. Returning to menu.\n")
            return False

        adapter = FaceRecognitionAdapter(self.faces_path)
        if not adapter.initialize():
            print("[ERROR] Face recognition model is not available. Ensure models are installed.\n")
            return False

        if adapter.has_identity(name):
            try:
                overwrite = self._input(f"Identity '{name}' is already enrolled. Overwrite? (y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                overwrite = "n"
            if overwrite != "y":
                print("[INFO] Registration cancelled by operator.\n")
                return False

        print(f"\n[INFO] Opening webcam (device {camera_index}) for enrollment...")
        print("[INFO] Instructions:")
        print("       - Look directly at camera with neutral expression.")
        print("       - Press [SPACE] to capture when face box turns GREEN.")
        print("       - Press [Q] or ESC to cancel.\n")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam device {camera_index}. Ensure camera is connected.\n")
            return False

        captured_emb = None
        consecutive_valid = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[ERROR] Failed to read frame from webcam.")
                    break

                emb, crop, status_msg, bbox = adapter.extract_face(frame)
                is_valid = (emb is not None)

                display_frame = frame.copy()
                if bbox:
                    x1, y1, x2, y2 = bbox
                    color = (0, 255, 0) if is_valid else (0, 165, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    status_text = "READY - PRESS SPACE" if is_valid else status_msg
                    cv2.putText(display_frame, status_text, (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(display_frame, status_msg, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(display_frame, f"Enrolling: {name} | [SPACE] Capture | [Q] Cancel",
                            (20, display_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                try:
                    cv2.imshow("Waste_dumping_v2 Face Registration", display_frame)
                    key = cv2.waitKey(30) & 0xFF
                except Exception:
                    key = 255

                if is_valid:
                    consecutive_valid += 1
                else:
                    consecutive_valid = 0

                if key == ord(' '):
                    if is_valid:
                        captured_emb = emb
                        print(f"\n[INFO] Valid face captured for '{name}'.")
                        break
                    else:
                        print(f"[WARNING] Cannot capture: {status_msg}")
                elif key == ord('q') or key == 27:
                    print("\n[INFO] Face registration cancelled by operator.")
                    break
                elif key == 255 and consecutive_valid >= 20:
                    # Headless auto-capture fallback after 20 consecutive solid frames
                    captured_emb = emb
                    print(f"\n[INFO] Auto-captured valid face for '{name}' (headless mode).")
                    break
        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if captured_emb is not None:
            success, msg = adapter.register_identity(name, captured_emb, replace=True)
            if success:
                print(f"[SUCCESS] {msg}")
                print(f"[INFO] Current enrolled identities: {', '.join(adapter.get_enrolled_names())}\n")
                return True
            else:
                print(f"[ERROR] Enrollment failed: {msg}\n")
                return False
        else:
            print("[INFO] No face was registered. Returning to menu.\n")
            return False

    # -------------------------------------------------------------------------
    # Option 2: Live Webcam Detection
    # -------------------------------------------------------------------------
    def webcam_detection_workflow(self) -> None:
        """Start detection pipeline from a local webcam device."""
        print("\n" + "=" * 65)
        print("          OPTION 2: LIVE WEBCAM DETECTION")
        print("=" * 65)
        available = probe_available_webcams(max_probe=3)
        if available:
            print(f"[INFO] Detected available camera index(es): {available}")
        else:
            print("[INFO] No active cameras enumerated; will attempt default index 0.")

        try:
            cam_input = self._input("Enter camera device index [default: 0]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Cancelled.")
            return

        cam_idx = cam_input if cam_input else "0"

        coordinator = ApplicationCoordinator(self.settings_path, self.zones_path)
        if not coordinator.initialize(source_override=str(cam_idx)):
            print(f"[ERROR] Failed to initialize pipeline for camera {cam_idx}. Returning to menu.\n")
            return

        print(f"\n[INFO] Starting live detection on webcam {cam_idx}...")
        print("[INFO] Press 'q' in the preview window or Ctrl+C in terminal to stop.\n")
        self._run_coordinator_stream(coordinator, window_title=f"Live Webcam Monitor (Device {cam_idx})")

    # -------------------------------------------------------------------------
    # Option 3: CCTV Camera Detection
    # -------------------------------------------------------------------------
    def cctv_detection_workflow(self) -> None:
        """Connect to an authorized RTSP/HTTP CCTV video stream."""
        print("\n" + "=" * 65)
        print("          OPTION 3: AUTHORIZED CCTV STREAM DETECTION")
        print("=" * 65)
        try:
            stream_url = self._input("Enter authorized CCTV stream URL (RTSP / HTTP): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Cancelled.")
            return

        if not stream_url:
            print("[INFO] Stream URL cannot be empty. Returning to menu.\n")
            return

        sanitized_url = sanitize_stream_url(stream_url)
        print(f"[INFO] Testing connection to {sanitized_url}...")

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"[ERROR] Connection failed: {sanitized_url}")
            print("[ERROR] Stream is offline, unreachable, or credentials rejected.\n")
            return

        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print(f"[ERROR] Connected to {sanitized_url}, but failed to read frames.\n")
            return

        print(f"[SUCCESS] Stream connection verified. Launching detection pipeline...")
        coordinator = ApplicationCoordinator(self.settings_path, self.zones_path)
        if not coordinator.initialize(source_override=stream_url):
            print(f"[ERROR] Failed to initialize CCTV pipeline. Returning to menu.\n")
            return

        self._run_coordinator_stream(coordinator, window_title="CCTV Stream Monitor")

    # -------------------------------------------------------------------------
    # Option 4: Video File Detection (Native Windows File Picker)
    # -------------------------------------------------------------------------
    def video_file_workflow(self) -> None:
        """Open native Windows file picker and process the selected video."""
        print("\n" + "=" * 65)
        print("          OPTION 4: VIDEO FILE DETECTION")
        print("=" * 65)
        print("[INFO] Opening native Windows file selection dialog...")

        video_path = self._file_picker()
        if not video_path:
            print("[INFO] File selection cancelled by operator. Returning to menu.\n")
            return

        p = Path(video_path)
        if not p.is_file():
            print(f"[ERROR] Selected path '{video_path}' does not exist or is not a file.\n")
            return

        if p.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            print(f"[ERROR] Unsupported video format '{p.suffix}'.")
            print(f"[INFO] Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}\n")
            return

        print(f"\n[INFO] Selected video: {video_path}")
        print("[INFO] Initializing detection pipeline for video processing...")

        coordinator = ApplicationCoordinator(self.settings_path, self.zones_path)
        if not coordinator.initialize(source_override=str(video_path)):
            print("[ERROR] Failed to initialize pipeline for video file. Returning to menu.\n")
            return

        self._run_coordinator_stream(coordinator, window_title=f"Video Monitor: {p.name}")

    # -------------------------------------------------------------------------
    # Option 5: View Event Records
    # -------------------------------------------------------------------------
    def view_records_workflow(self) -> None:
        """Read and format recorded events from data/events.csv."""
        print("\n" + "=" * 65)
        print("          OPTION 5: RECORDED EVENT LOG")
        print("=" * 65)
        storage = CSVEventStorage(self.csv_path)
        records = storage.get_all_records()

        if not records:
            print("[INFO] No event records found in data/events.csv.\n")
        else:
            print(f"[INFO] Total recorded events: {len(records)}\n")
            print(f"{'Event ID':<26} {'Confirmed At':<20} {'Type':<22} {'Category':<12} {'Person':<10} {'Face':<12} {'Attribution'}")
            print("-" * 115)
            for r in records:
                eid = r.get("event_id", "")
                ts = r.get("confirmed_at", "")[:19]
                etype = r.get("event_type", "")
                cat = r.get("waste_category", "")
                pid = r.get("person_track_id", "")
                face = r.get("recognized_person", "")
                attr = r.get("attribution_status", "")
                print(f"{eid:<26} {ts:<20} {etype:<22} {cat:<12} {pid:<10} {face:<12} {attr}")
            print("=" * 115 + "\n")

        try:
            self._input("Press Enter to return to main menu...")
        except (EOFError, KeyboardInterrupt):
            pass

    # -------------------------------------------------------------------------
    # Option 6: Exit
    # -------------------------------------------------------------------------
    def exit_workflow(self) -> bool:
        """Exit the application cleanly."""
        print("\n[INFO] Exiting Waste Dumping Detection System. Goodbye!\n")
        return False

    # -------------------------------------------------------------------------
    # Internal Pipeline Execution Helper
    # -------------------------------------------------------------------------
    def _run_coordinator_stream(
        self,
        coordinator: ApplicationCoordinator,
        window_title: str = "Waste_dumping_v2 Live Monitor",
    ) -> None:
        """Execute coordinator loop with console logging and optional preview."""
        last_log_time = time.time()

        def frame_callback(frame, persons, waste, events):
            nonlocal last_log_time
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
                    print(f" Event ID:           {ev.event_id}")
                    print(f" Responsible Person: {ev.person_track_id} ({ev.attribution_status.value})")
                    print(f" Recognized Face:    {ev.recognized_person} ({ev.recognition_status.value})")
                    print(f" Waste Track:        {ev.waste_track_id} ({ev.waste_category}, conf={ev.waste_confidence:.2f})")
                    print(f" Zone:               {ev.ground_zone_id}")
                    print("!" * 70 + "\n")

            for p in persons:
                x1, y1, x2, y2 = p.bbox.to_xyxy_int()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, p.track_id, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            for w in waste:
                x1, y1, x2, y2 = w.bbox.to_xyxy_int()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{w.track_id} ({w.class_name})", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            try:
                cv2.imshow(window_title, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    coordinator.stop()
            except Exception:
                pass

        try:
            coordinator.run(callback=frame_callback)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped by operator interrupt.")
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        print("\n" + "=" * 70)
        print(" Session Complete.")
        print(f" Total Frames Processed: {coordinator.frame_count}")
        print(f" Confirmed Events:       {coordinator.total_events}")
        print(f" Records Saved in CSV:   {coordinator.storage.total_records}")
        print("=" * 70 + "\n")
