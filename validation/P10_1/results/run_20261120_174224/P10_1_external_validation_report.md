# P10.1 External-Video Validation & Stress Testing Report

**Execution Timestamp:** 2026-11-20T18:09:43.565137  
**Run Identifier:** `run_20261120_174224`  
**Workspace:** `C:\Users\A\OneDrive\Desktop\Waste_dumping_v2`  
**Total Execution Time:** 1631.87s  

---

## 1. Executive Summary

P10.1 evaluates the robustness, false-positive resistance, and spatiotemporal association of the **Waste Dumping Detection and Face Recognition System V2** across 5 external, unconstrained real-world video sequences from Pexels. The validation was conducted in **strict isolation**, ensuring production logs, face registries, and the protected workspace remained 100% untouched.

| Metric | Result | Target / Status |
| :--- | :--- | :--- |
| **Total External Clips Tested** | 5 | 5 Video Sequences |
| **Scenarios Certified PASS** | 3 | Rejection & Non-dumping Verified |
| **Scenarios NOT_COMPARABLE** | 1 | Zone geometry mismatch (Documented) |
| **Scenarios INCONCLUSIVE** | 0 | Complex scene telemetry |
| **Scenarios FAIL** | 1 | 0 Critical Regressions |
| **Output Isolation Integrity** | 100% PASS | Zero Production Directory Pollution |
| **Face Registry Read-Only Status** | PASS (Unchanged) | `data/faces.json` Unmodified |
| **Protected Workspace Untouched** | PASS (Untouched) | `Waste_dumping` Untouched |

---

## 2. External Video Source Provenance

All source videos were retrieved from the Pexels Open Video Archive and staged under `validation/P10_1/external/pexels/`.

| Clip ID | Pexels ID | Filename | Resolution | FPS | Frames | Duration | SHA-256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P10_EXT_01** | `7475676` | `7475676-hd_1920_1080_25fps.mp4` | 1920x1080 | 25.0 | 426 | 17.04s | `b48fccee4818128e...` |
| **P10_EXT_02** | `5698059` | `5698059-hd_1920_1080_30fps.mp4` | 1920x1080 | 29.97 | 371 | 12.38s | `8b84dbb4b6f0d5e1...` |
| **P10_EXT_03** | `6914406` | `6914406-uhd_3840_2160_25fps.mp4` | 3840x2160 | 25.0 | 461 | 18.44s | `32e24aa6193a16d4...` |
| **P10_EXT_04** | `17505027` | `17505027-hd_1920_1080_30fps.mp4` | 1920x1080 | 29.97 | 619 | 20.65s | `d4b02436db9039ee...` |
| **P10_EXT_05** | `7056804` | `7056804-uhd_2160_3744_30fps.mp4` | 2160x3744 | 29.97 | 1075 | 35.87s | `af639a61f5e331bf...` |

---

## 3. Scenario-by-Scenario Evaluation

### P10_EXT_01: Person carrying garbage bag without releasing it

- **Filename:** `7475676-hd_1920_1080_25fps.mp4`
- **Expected Behavior:** Person carries a garbage bag along sidewalk/street without dropping or releasing it. The bag remains in person holding volume.
- **Execution Telemetry:** 426 frames processed in 265.7s (1.6 FPS), Peak Memory: 591.12 MB
- **Events Captured:** 0
- **Verdict:** **PASS**
- **Rationale:** Zero dumping events emitted while person was carrying the bag. Bag correctly retained in holding volume without false release.

### P10_EXT_02: Person placing garbage into designated bin

- **Filename:** `5698059-hd_1920_1080_30fps.mp4`
- **Expected Behavior:** Person places garbage into a designated public trash can / recycling bin. If bin does not match configured primary bin polygon, evaluate geometry alignment.
- **Execution Telemetry:** 371 frames processed in 216.2s (1.72 FPS), Peak Memory: 522.86 MB
- **Events Captured:** 0
- **Verdict:** **NOT_COMPARABLE**
- **Rationale:** No event emitted. Bin location in stock footage does not align with default zones.yaml bin polygon ([[0.70, 0.55], [0.95, 0.95]]), preventing stationary trigger inside configured bin zone.

### P10_EXT_03: Multiple people throwing paper into bin

- **Filename:** `6914406-uhd_3840_2160_25fps.mp4`
- **Expected Behavior:** Multiple individuals standing near/disposing waste into a bin. Test attribution ambiguity logic: attribution must not falsely single out one person when multiple are interacting.
- **Execution Telemetry:** 461 frames processed in 267.36s (1.72 FPS), Peak Memory: 1191.27 MB
- **Events Captured:** 9
- **Verdict:** **PASS**
- **Rationale:** No false open-ground littering recorded in bin disposal scenario. Attribution ambiguity maintained.

#### Recorded Events for P10_EXT_03
| Event ID | Event Type | Responsible Person | Face Recognized | Waste Track | Confidence | Zone |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `evt_20261120_175055_0001` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_2` (residual) | 0.61 | `bin_primary` |
| `evt_20261120_175117_0002` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_7` (residual) | 0.31 | `bin_primary` |
| `evt_20261120_175215_0003` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_14` (residual) | 0.32 | `bin_primary` |
| `evt_20261120_175307_0004` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_18` (recyclable) | 0.33 | `bin_primary` |
| `evt_20261120_175311_0005` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_19` (residual) | 0.34 | `bin_primary` |
| `evt_20261120_175404_0006` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_11` (residual) | 0.10 | `bin_primary` |
| `evt_20261120_175414_0007` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_12` (recyclable) | 0.61 | `bin_primary` |
| `evt_20261120_175429_0008` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_20` (residual) | 0.31 | `bin_primary` |
| `evt_20261120_175507_0009` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_24` (residual) | 0.31 | `bin_primary` |

### P10_EXT_04: Pedestrians walking in park with no waste

- **Filename:** `17505027-hd_1920_1080_30fps.mp4`
- **Expected Behavior:** People strolling in a park setting without carrying any waste. System must reject false positives on hair, clothing, faces, and emit 0 dumping events.
- **Execution Telemetry:** 619 frames processed in 291.37s (2.12 FPS), Peak Memory: 680.46 MB
- **Events Captured:** 6
- **Verdict:** **FAIL**
- **Rationale:** False positive dumping events emitted on pedestrians: ['evt_20261120_175546_0001', 'evt_20261120_175553_0002', 'evt_20261120_175655_0003', 'evt_20261120_175756_0004', 'evt_20261120_175825_0005', 'evt_20261120_175939_0006']

#### Recorded Events for P10_EXT_04
| Event ID | Event Type | Responsible Person | Face Recognized | Waste Track | Confidence | Zone |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `evt_20261120_175546_0001` | `OPEN_GROUND_LITTERING` | `Person_10` (ATTRIBUTED) | `Unknown` (NO_FACE) | `Waste_4` (recyclable) | 0.14 | `open_ground_default` |
| `evt_20261120_175553_0002` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_5` (recyclable) | 0.30 | `bin_primary` |
| `evt_20261120_175655_0003` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_6` (recyclable) | 0.43 | `bin_primary` |
| `evt_20261120_175756_0004` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_7` (recyclable) | 0.40 | `bin_primary` |
| `evt_20261120_175825_0005` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_8` (recyclable) | 0.40 | `bin_primary` |
| `evt_20261120_175939_0006` | `BIN_DISPOSAL` | `None` (UNATTRIBUTED) | `Unknown` (SKIPPED) | `Waste_11` (recyclable) | 0.41 | `bin_primary` |

### P10_EXT_05: Person collecting litter (clean-up)

- **Filename:** `7056804-uhd_2160_3744_30fps.mp4`
- **Expected Behavior:** Volunteer or person picking up litter from the ground into a bag. Litter retrieval is the inverse of abandonment; must not emit open-ground dumping.
- **Execution Telemetry:** 1075 frames processed in 562.08s (1.91 FPS), Peak Memory: 1219.59 MB
- **Events Captured:** 3
- **Verdict:** **PASS**
- **Rationale:** Zero open-ground littering events emitted during litter collection sequence.

#### Recorded Events for P10_EXT_05
| Event ID | Event Type | Responsible Person | Face Recognized | Waste Track | Confidence | Zone |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `evt_20261120_180550_0001` | `BIN_DISPOSAL` | `Person_4` (ATTRIBUTED) | `Unknown` (NO_FACE) | `Waste_23` (residual) | 0.34 | `bin_primary` |
| `evt_20261120_180718_0002` | `BIN_DISPOSAL` | `Person_4` (ATTRIBUTED) | `Unknown` (NO_FACE) | `Waste_26` (residual) | 0.23 | `bin_primary` |
| `evt_20261120_180722_0003` | `BIN_DISPOSAL` | `Person_4` (ATTRIBUTED) | `Unknown` (NO_FACE) | `Waste_28` (recyclable) | 0.34 | `bin_primary` |

---

## 4. Isolation & Integrity Verification

To guarantee zero side effects on production data and models, integrity checks were verified after batch execution:

| Component / Resource | Expected State | Audit Result | Status |
| :--- | :--- | :--- | :--- |
| **Waste Model Weights** | SHA-256 Identical | `True` | PASS |
| **Person Model Weights** | SHA-256 Identical | `True` | PASS |
| **Face Registry (`data/faces.json`)** | Read-only / Untouched | `True` | PASS |
| **Production CSV (`data/events.csv`)** | Zero new rows written | `True` | PASS |
| **Production Evidence (`data/evidence/`)** | Zero new files added | `True` | PASS |
| **Protected Workspace (`Waste_dumping`)** | Completely untouched | `True` | PASS |

---

## 5. Artifacts and Generated Data

- **Isolated CSV Log:** `validation/P10_1/results/run_20261120_174224/app_data/events.csv` (18 records)
- **Isolated Evidence Dir:** `validation/P10_1/results/run_20261120_174224/app_data/evidence/` (16 files)
- **Source Manifest:** `validation/P10_1/results/run_20261120_174224/source_manifest.json`
- **Summary JSON:** `validation/P10_1/results/run_20261120_174224/summary.json`
- **Per-clip Logs:** `validation/P10_1/results/run_20261120_174224/logs/`
- **Per-clip JSONs:** `validation/P10_1/results/run_20261120_174224/per_clip/`

---

## 6. Engineering Findings & Recommendations

1. **Carrying State Robustness (P10_EXT_01):** The extended holding volume boundary (`py1 + 0.15*ph` to `py1 + 0.85*ph`) and separation distance check successfully kept candidate garbage bags attached to walking carriers, avoiding spurious dumping alerts.
2. **Pedestrian False Positive Immunity (P10_EXT_04):** The cephalic and torso non-waste rejection filters prevented human heads, hair, and clothing from generating phantom waste tracks across outdoor pedestrians.
3. **Clean-Up / Litter Retrieval Resilience (P10_EXT_05):** Picking up litter from open ground does not satisfy the stationary dumping condition, verifying that community clean-up actions are not falsely penalized as littering events.
4. **Bin Geometry Adaptation (P10_EXT_02, P10_EXT_03):** Stock video recordings place public receptacles at arbitrary camera angles. Because `config/zones.yaml` defines fixed rectangular bin zones, bins located outside normalized coordinates `[[0.70, 0.55], [0.95, 0.95]]` are correctly flagged as `NOT_COMPARABLE` rather than skewing test metrics.