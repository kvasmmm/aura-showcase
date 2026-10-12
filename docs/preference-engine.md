# Preference Engine & Recommendation Architecture

> **Source Reference:** Aura backend revision `f713a18` (`backend/app/services/preference_engine.py`, `backend/app/services/recommendation_service.py`, `backend/app/services/active_learning_service.py`, `backend/app/ml/recommenders/`).

---

## 1. Feature Representation & Normalization

Aura operates on a **10-dimensional continuous geometric feature space** defined in `backend/app/core/features.py`:

| Feature Name | Display Name | Status | Description |
|---|---|---|---|
| `intercanthal_to_face_ratio` | Eye spacing | Stable | Distance between inner eye corners normalized by face width |
| `eye_width_to_face_ratio` | Eye width | Stable | Mean width of eye openings normalized by face width |
| `nose_width_to_face_ratio` | Nose width | Stable | Alar base width normalized by face width |
| `mouth_width_to_face_ratio` | Mouth width | Stable | Distance between lip corners normalized by face width |
| `lower_lip_to_mouth_ratio` | Lower lip fullness | Stable | Height of lower vermilion normalized by mouth width |
| `canthal_tilt_deg` | Eye tilt | Experimental | Angle between inner and outer canthi in degrees |
| `nose_bridge_width_ratio` | Nose bridge width | Experimental | Width of nasal root normalized by face width |
| `face_height_to_width` | Face elongation | Experimental | Tragus-to-menton vs bizygomatic proportion |
| `chin_taper_angle_deg` | Chin taper | Experimental | Jawline convergence angle towards menton |
| `brow_arch_ratio_v2` | Brow arch | Experimental | Peak brow apex elevation relative to eye plane |

The 5 "stable" features were established during preliminary repeatability screening on LFW identities; the 5 "experimental" features are included in runtime scoring models.

### Normalization
At startup, `PoolScaler` (`backend/app/core/scaler.py`) computes the median and Interquartile Range (IQR) for each feature across `pool.csv`:
$$\tilde{x}_i = \text{clamp}\left(\frac{x_i - \text{median}_i}{\text{IQR}_i}, -3.0, 3.0\right)$$
All downstream dot products, EMA centroids, and FM sparse inputs operate on these normalized coordinates $\tilde{x}_i$.

---

## 2. Interaction Telemetry & Confidence Weighting

Each swipe event contains explicit feedback (`action \in {"like", "dislike"}`) and telemetry metrics captured client-side. The function `compute_confidence(event)` calculates a behavioral confidence scalar $c \in [0.05, 2.5]$:

```
Base Confidence = 1.0

Latency Rules:
- Like with latency < 120 ms:              × 0.20  (penalize accidental swipe)
- Dislike with latency < 200 ms:           × 1.50  (visceral rejection bonus)
- Fast like (200–800 ms, reversals=0):     × 1.50  (decisive preference bonus)
- Fast dislike (200–900 ms):               × 2.00  (decisive rejection bonus)
- Latency > 15,000 ms:                     × 0.50  (distraction penalty)

Hesitation Rules:
- hesitationScore > 0.50:                  × 0.60
- hesitationScore > 0.20:                  × 0.80

Reversal Rules:
- directionReversals >= 2:                 × 0.30
- directionReversals == 1:                 × 0.70

Zoom Bonus:
- zoomCount > 0:                           + 0.20

Final Step:
Confidence = clamp(Base Confidence × confidenceMultiplier, 0.05, 2.50)
```

---

## 3. Long-Term Preference Profile (EMA Centroids)

The long-term profile maintains running Exponential Moving Average (EMA) centroids for liked and disliked cards:

### Update Rules
For each event, sample weight $w$ is defined as:
$$w = w_{\text{base}} \times c, \quad \text{where } w_{\text{base}} = \begin{cases} 1.5 & \text{if action = dislike} \\ 1.0 & \text{if action = like} \end{cases}$$

The effective EMA smoothing factor is modulated by behavioral confidence:
$$\alpha_{\text{eff}} = \min(1.0, \alpha_{\text{base}} \times c), \quad \text{with } \alpha_{\text{base}} = 0.10$$

Centroid updates:
$$\mathbf{c}_{\text{liked}}^{(t)} = (1 - \alpha_{\text{eff}}) \mathbf{c}_{\text{liked}}^{(t-1)} + \alpha_{\text{eff}} \mathbf{x}^{(t)} \quad (\text{on like})$$
$$\mathbf{c}_{\text{disliked}}^{(t)} = (1 - \alpha_{\text{eff}}) \mathbf{c}_{\text{disliked}}^{(t-1)} + \alpha_{\text{eff}} \mathbf{x}^{(t)} \quad (\text{on dislike})$$

The overall user preference vector is the signed difference between centroids:
$$\mathbf{u} = \mathbf{c}_{\text{liked}} - \mathbf{c}_{\text{disliked}}$$

### Profile Stability
The confidence/maturity of the profile is quantified as:
$$\text{stability} = \min\left(1.0, \sqrt{\frac{\min(\text{likeMass}, \text{dislikeMass})}{20.0}}\right)$$

### Confirmed Negative Regions
Extracted from the user's last 20 dislikes: features exhibiting high dislike density with evidence counts $\ge 3$ are flagged as `confirmedNegativeRegions`.

---

## 4. Short-Term Session Preference Adaptation

To track intra-session preference shifts without polluting the long-term profile, Aura computes two parallel vectors over the recent interaction buffer:

1. **Short Vector ($\mathbf{s}$):** Exponentially decayed over time with half-life parameter $\tau = 300$ seconds:
   $$\mathbf{s} = \text{normalize}\left(\sum_{j} \exp\left(-\frac{\Delta t_j}{\tau}\right) \cdot w_{\text{action}}(a_j) \cdot \mathbf{x}_j\right)$$
2. **Long Vector ($\mathbf{l}$):** Normalized unweighted sum over the session replay buffer:
   $$\mathbf{l} = \text{normalize}\left(\sum_{j} w_{\text{action}}(a_j) \cdot \mathbf{x}_j\right)$$

### Dynamic Gate ($\alpha_{\text{gate}}$)
Calculated via `compute_alpha()` in `backend/app/ml/session_adaptation/dynamic_gate.py`:
- **Divergence:** $D = \text{clamp}(1.0 - \cos(\mathbf{s}, \mathbf{l}), 0.0, 2.0)$
- **Speed Score:** $S = \text{clamp}\left(\frac{4.0 - \text{avgGap}}{4.0}, 0.0, 1.0\right)$
- **Coherence:** $C = \text{clamp}(\text{mean pairwise cosine between recent likes}, 0.0, 1.0)$
- **Session Confidence:** $K = \min(1.0, N_{\text{events}} / 10.0)$

$$\alpha_{\text{raw}} = 0.15 + 0.45 D + 0.25 S + 0.25 C$$
$$\alpha_{\text{gate}} = \text{clamp}(\alpha_{\text{raw}} \times K, 0.10, 0.90)$$
*(Capped at $0.35$ if $C < 0.25$ or $D < 0.20$. Defaults to $0.15$ if $N_{\text{events}} < 3$.)*

The blended session user vector is:
$$\mathbf{u}_{\text{session}} = \text{normalize}((1 - \alpha_{\text{gate}}) \mathbf{l} + \alpha_{\text{gate}} \mathbf{s})$$

Card session score:
$$\text{score}_{\text{session}} = \cos(\mathbf{u}_{\text{session}}, \mathbf{x}) - 0.30 \cdot \max_{d \in \text{recent dislikes}} \cos(\mathbf{x}, \mathbf{d}) + 0.05 \cdot \text{novelty}$$

---

## 5. Dual Factorization Architecture: FM vs. MF

Aura implements two distinct factorization models serving separate stages of the pipeline:

| Dimension | Feature-based Factorization Machine (FM) | ID-based Matrix Factorization (MF) |
|---|---|---|
| **Class** | `FactorizationMachine` | `MatrixFactorizationRecommender` |
| **Source File** | `app/ml/recommenders/factorization_machine.py` | `app/ml/recommenders/matrix_factorization.py` |
| **Model Type** | 2-way sparse Factorization Machine (Rendle 2010) | Biased Matrix Factorization (Funk-MF / SVD) |
| **Mathematical Formulation** | $\hat{y} = w_0 + \sum_{i=1}^{p} w_i x_i + \sum_{i=1}^{p} \sum_{j=i+1}^{p} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$ | $\hat{r}_{u,i} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$ |
| **Latent Factors ($k$)** | **$k = 4$** (`FM_K_LATENT_FACTORS = 4`) | **$k = 8$** (hardcoded in `active_learning_service.py`) |
| **Input Data** | 10 continuous facial geometry features ($x_i \in \mathbb{R}$) | User ID string ($u$), Card ID string ($i$) |
| **Target Variable** | Binary: $+1$ (like), $-1$ (dislike) | Rating: $5.0$ (like), $1.0$ (dislike/report) |
| **Loss Function** | Log-loss on margin: $\ln(1 + \exp(-y \cdot \hat{y}))$ | Squared error: $\frac{1}{2}(r_{u,i} - \hat{r}_{u,i})^2$ |
| **Training Trigger** | Async `BackgroundTask` after `/telemetry` | Synchronous on **every `GET /recommendations` call** |
| **Preconditions** | $\text{likes} \ge 3$, $\text{dislikes} \ge 3$, $\text{events} \ge 6$ | Non-empty `active_learning_ratings` ($\le 5000$ rows) |
| **Persistence** | **Persisted**: Saved in PostgreSQL snapshot | **Transient**: In-memory fit and discard |
| **Pipeline Role** | Directly modulates candidate ranking `finalScore` | Feeds `VotingStrategy` for active-learning exploration |
| **Linear Time Evaluation** | Yes: evaluates pairwise interactions in $O(k \cdot \text{nnz})$ | Yes: dot product of length $k$ |

---

## 6. Candidate Scoring & Ranking Formula

When candidate cards are scored in `RecommendationService.calculate_score()`:

$$\text{finalScore} = \text{baseScore} + \text{affinityBonus} + (\text{fmScore} \times \text{fmAmp}) + (\text{sessionScore} \times \text{sessionAmp}) - \text{negativePenalty}$$

### Components:
1. **Base Score:** Linear dot product between user vector and card vector:
   $$\text{baseScore} = \mathbf{u}^T \mathbf{x}$$
2. **Affinity Bonus:** Non-linear boost applied if candidate exhibits extreme values on the user's top positive features.
3. **FM Score:**
   $$\text{fmScore} = \text{clamp}(\hat{y}_{\text{FM}}, -5.0, 5.0) \times 3.5$$
   *(If FM is not yet trained and updates $\le 50$, falls back to cosine similarity).*
4. **Session Coherence Amplifiers:**
   $$\text{fmAmp} = 1.0 + 1.5 \cdot C$$
   $$\text{sessionAmp} = 1.0 + 2.0 \cdot C$$
   *(Where $C$ is recent like coherence. Amplifies model influence when the user is decisively focused).*
5. **Confirmed Negative Penalty:** If card feature values match a confirmed negative region:
   $$\text{penalty} = 7.0 \times \text{negConf} \times (1.0 + |u_i|) \times \max(\text{intensity}, 0.2)$$
   Cards exceeding strict negative thresholds are flagged as `isHardNegative`.

---

## 7. Recommendation Batch Composition

Batches (default limit = 20) are assembled through a multi-tier allocation:

```
[Candidate Pool: 98 cards]
  │
  ├─► Filter seen and excluded cards
  │
  ├─► Partition candidates:
  │     ├── normal_pool: cards with no hard-negative conflicts
  │     └── penalty_pool: cards matching active confirmed negative regions
  │
  ├─► Active Learning Exploration (up to ~25% of batch):
  │     - Fits transient ID-based MF (k=8, 12 epochs) on multi-user history
  │     - Evaluates VotingStrategy([HighestPredicted, Variance, MinNorm])
  │     - Pulls top exploratory cards from normal_pool
  │
  ├─► Heuristic Exploration Buckets:
  │     - Boundary Test (~2%): probes moderate negative penalty [0.4, 1.6]
  │     - Uncertain Feature Probe (<=8%): probes cards with |baseScore| < 0.2, no penalty
  │     - Promising Exploration (<=12%): probes cards with baseScore > 0.2, no penalty
  │
  ├─► Exploitation Bucket:
  │     - Fills remaining slots with highest finalScore cards from normal_pool
  │
  └─► Exhaustion Fallback:
        - If normal_pool is depleted, recycles high-scoring seen cards or relaxed penalty pool
```

---

## 8. Implemented vs. Inactive Algorithms Audit

| Algorithm / Module | Status | Runtime Reality |
|---|---|---|
| `FactorizationMachine` ($k=4$, SGD) | **Active** | Trained on user history replay buffer; modulates ranking score. |
| `MatrixFactorizationRecommender` ($k=8$) | **Active** | Refit on every recommendation request; powers AL candidate selection. |
| `VotingStrategy` Ensemble | **Active** | Combines `HighestPredicted`, `Variance`, and `MinNorm`. |
| `EntropyStrategy` (`entropy.py`) | **Unused** | Calculates Shannon entropy over rating distributions; not mounted in runtime engine. |
| `HELFStrategy` (`helf.py`) | **Unused** | Harmonic mean of entropy and log-frequency; unmounted. |
| `Entropy0Strategy` (`entropy0.py`) | **Unused** | Zero-imputed entropy; unmounted. |
| `AdaptiveSwitchingStrategy` | **Unused** | Exploration/exploitation threshold switcher; unmounted. |
| Variance Hunters in AL Selector | **Bypassed** | Code exists for 70/30 exploitation-variance split; bypassed because `RecommendationService` calls selector without score metadata. |
| Demographic Hard Veto | **Stub** | Stub function returning input unchanged (`return candidate_ids`). |
| Long-Vector EMA (`update_long_vector_ema`) | **Unused** | Alternative online EMA; runtime uses historical sum. |
