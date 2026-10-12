# Data Pipeline (Aura Face Profiler)

> **Source Reference:** Aura Face Profiler repository revision `593f4e2` (`aura-face-profiler/`).  
> **Role:** Decoupled, offline computer vision and geometry extraction pipeline. It is not on the runtime web request path.

---

## 1. Computer Vision & Landmark Extraction

The Face Profiler processes raw portrait images into geometric feature vectors:

1. **Landmark Detection:** Utilizes MediaPipe Face Landmarker to extract 468 3D facial landmarks.
2. **Quality Gate:**
   - Detects exactly one face (rejects images with 0 or multiple faces).
   - Enforces a minimum face-to-image size ratio of $0.04$.
   - Validates border safety margins (margin $\ge 0.015$).
   - Enforces strict head-pose limits:
     - Yaw: $\le 25^\circ$
     - Pitch: $\le 20^\circ$
     - Roll: $\le 45^\circ$
3. **Head Pose Estimation:** Extracts head rotation from MediaPipe's $4 \times 4$ transformation matrix decomposed via `cv2.RQDecomp3x3`, with a 2D landmark geometric heuristic fallback.
4. **Canonical Alignment:** Computes an affine partial similarity transform (`cv2.estimateAffinePartial2D`) from eye and nose landmarks to project faces into a standardized $512 \times 512$ canonical coordinate space.

---

## 2. Geometric Feature Space

Aura extracts continuous geometric ratios invariant to scale and translation:

### 5 Stable Features (Repeatability Verified)
Selected via preliminary repeatability evaluation on 15 identities (77 photos, 68 passed gate) from the LFW dataset:
- `intercanthal_to_face_ratio`
- `eye_width_to_face_ratio`
- `nose_width_to_face_ratio`
- `mouth_width_to_face_ratio`
- `lower_lip_to_mouth_ratio`

### 5 Experimental Features
Retained for candidate modeling and exploration:
- `canthal_tilt_deg`
- `nose_bridge_width_ratio`
- `face_height_to_width`
- `chin_taper_angle_deg`
- `brow_arch_ratio_v2`

---

## 3. Catalog Selection & Batch Ingestion

To build the runtime candidate catalog (`pool.csv`):
1. Images are scored deterministically by quality:
   $$\text{score} = 0.01 \cdot \min(\text{blur}, 500) + 10 \cdot \text{faceSize} - (2|\text{yaw}| + 1.5|\text{pitch}| + 1.0|\text{roll}|)$$
2. The highest-scoring single photo per identity is selected.
3. The resulting catalog contains **98 unique identities**.

### Batch Run Verification
From the recorded batch ingestion run (`kpop_faces_clean/summary.json`):
- **Total Images Processed:** 4,482
- **Accepted Images:** 3,849
- **Rejected Images:** 633 (581 face out of frame, 25 pitch excess, 22 yaw excess, 4 multiple faces, 1 no face)
- **Quality-Gate Pass Rate:** 85.88% (reflects input photo suitability against pose/cropping criteria, not model classification accuracy)
- **Runtime:** 257.84 seconds at concurrency 4 (~17.38 images/sec; recorded in summary file)

---

## 4. Population Calibration (Deferred)

The Face Profiler repository contains prototyping code for population-level Z-score and hyperbolic tangent normalization (`tanh(z / 2.5)` based on a 1,000-image cohort). In the current runtime catalog, this calibration is **disabled** (`normalized_features=None`, `calibration_version=None`); normalization is performed on the Aura backend at startup via median/IQR scaling (`PoolScaler`).

---

## 5. Data Rights & Distribution

Candidate photos represent public figures. Because commercial licensing and redistributable rights are unverified, dataset images and raw photo files are strictly excluded from public distribution.
