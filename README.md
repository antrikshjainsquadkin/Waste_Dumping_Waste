# Waste Detection & Face Recognition System (V2)

An integrated, modular computer vision pipeline for automated waste disposal and open-ground littering detection with event-triggered facial recognition offender identification.

## Core Architecture

```
Camera Stream
     │
     ▼
[SharedCameraService] ──> [RollingFrameBuffer (12s Ring Buffer)]
     │
     ├──> [PersonDetector (YOLO11 COCO)] ──> Person_1, Person_2, ...
     │                                              │
     └──> [WasteDetector (YOLO11m khanhromvn)] ─────┼──> 3-Phase Non-Waste Rejection Gate
                                                    │    (Rejects limbs, body, clothes)
                                                    ▼
                                     [MultiObjectTrackingService]
                                     (ByteTrack Person & Waste IDs)
                                                    │
                                                    ▼
                                   [PersonWasteAssociationEngine]
                                 (Kinematic Spatiotemporal Tracking)
                                                    │
                                        ┌───────────┴───────────┐
                                        ▼                       ▼
                                 [Bin Disposal]        [Open-Ground Littering]
                                 (Receptacle polygon)  (Stability ≥3-5s & Separation)
                                        │                       │
                                        └───────────┬───────────┘
                                                    │
                                                    ▼
                                       [ConfirmedLitterEvent]
                                                    │
                                                    ├──> [FaceRecognitionAdapter]
                                                    │    (InsightFace buffalo_s)
                                                    │    (Runs ONLY on confirmed events)
                                                    │
                                                    ├──> [EvidenceRecorder]
                                                    │    (5s Pre + 5s Post MP4 Video Clip)
                                                    │    (High-Quality Face Snapshot JPEG)
                                                    │
                                                    └──> [CSVEventStorage]
                                                         (data/events.csv)
```

## Features

1. **Open-Ground Littering Detection**: Identifies waste tossed onto open ground, roads, sidewalks, or monitored outdoor areas without requiring a dustbin.
2. **Receptacle Bin Disposal**: Retains explicit bin disposal confirmation when waste enters a designated receptacle polygon.
3. **Dedicated Detectors**:
   - Waste: Pretrained Hugging Face YOLO11m checkpoint (`khanhromvn/waste-detection-yolo11`) detecting `organic`, `recyclable`, and `residual`.
   - Person: Dedicated YOLO11 COCO person detector (`person_detector.pt`).
4. **False Detection Rejection**: 3-Phase filter eliminates human hands, limbs, clothing, and full body false positive waste detections.
5. **Robust Person Attribution**: Evaluates spatial interaction history to attribute the disposal to the actual depositor rather than accidental bystanders.
6. **Pre-Existing Litter Rejection**: Pre-existing stationary litter in the scene emits zero false events.
7. **Carrying Without Disposal Rejection**: Carrying waste without dropping or releasing emits zero false events.
8. **Decoupled Facial Recognition**: InsightFace runs exclusively upon confirmed disposal, identifying enrolled individuals from `data/faces.json` or recording `Unknown`.
9. **5s Pre / 5s Post Video Evidence**: Generates a synchronized 10-second MP4 evidence clip centered on the estimated release time.
10. **Zero External Databases**: Completely local, CPU-compatible, lightweight, terminal-based application.

## Quickstart

### 1. Run Live Webcam
```bash
python main.py --source 0
```

### 2. Run on Recorded Video File
```bash
python main.py --source "path/to/video.mp4"
```

### 3. Run Headless (Background / Server)
```bash
python main.py --source "path/to/video.mp4" --no-preview
```

### 4. Custom Zones Configuration
Edit `config/zones.yaml` to specify ground area polygons and bin coordinates.
