# Objective 2: Soft-DTW Loss

**Branch:** `devBranch-offset-loss-dtw`
**Loss function:** Soft-DTW (γ = 1.0)
**Models:** LSTM and PatchTST (identical architecture and Optuna tuning as baseline)

---

## 1. Loss Function

Soft-DTW is a differentiable relaxation of Dynamic Time Warping. Instead of the hard minimum
in the DTW recursion, Soft-DTW uses a soft-minimum via the log-sum-exp operator controlled by
smoothing parameter γ:

```
soft-min_γ(a, b, c) = -γ · log(e^{-a/γ} + e^{-b/γ} + e^{-c/γ})
```

As γ → 0, Soft-DTW recovers standard DTW. For γ = 1.0 the loss provides a smooth gradient
signal throughout training. Unlike bounded-lag MSE, which assumes a global constant shift,
Soft-DTW allows non-linear, sample-wise temporal warping, permitting the model to align
predicted and true trajectories more flexibly.

The computational cost is O(H²) per sample, where H = 12 is the prediction horizon. Evaluation
remains strict pointwise RMSE at h = 11 (60 min), preserving a consistent train–eval mismatch
design across all three objectives.

---

## 2. Hyperparameters (Optuna, 50 trials, patient 85202)

| Parameter | LSTM | PatchTST |
|---|---|---|
| hidden_size / d_model | 256 | 256 |
| num_layers / n_layers | 2 | 3 |
| dropout | 0.038 | 0.140 |
| learning rate | 9.96e-4 | 6.53e-4 |
| batch_size | 512 | 256 |
| dim_ff | — | 128 |
| n_heads | — | 4 |

---

## 3. Summary Results vs Baseline

| Model | RMSE | Δ vs baseline | MAE | Precision | Recall | F1 | ΔF1 vs baseline |
|---|---|---|---|---|---|---|---|
| LSTM MSE (baseline) | 36.089 | — | 27.118 | 0.0104 | 0.0036 | 0.0054 | — |
| **LSTM Soft-DTW** | **37.396** | **+3.6%** | **29.110** | 0.0435 | 0.0125 | 0.0153 | +183% |
| PatchTST MSE (baseline) | 39.271 | — | 29.001 | 0.0482 | 0.0888 | 0.0609 | — |
| **PatchTST Soft-DTW** | **43.453** | **+10.6%** | **32.158** | 0.0766 | 0.1586 | 0.0835 | +37% |

---

## 4. Per-Patient Results — LSTM Soft-DTW

| Patient | RMSE | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 85101 | 21.125 | 16.930 | 0.000 | 0.000 | 0.000 |
| 85102 | 46.962 | 38.314 | 0.000 | 0.000 | 0.000 |
| 85104 | 27.996 | 22.099 | 0.000 | 0.000 | 0.000 |
| 85105 | 33.889 | 21.966 | 0.455 | 0.217 | 0.294 |
| 85106 | 32.189 | 23.841 | 0.000 | 0.000 | 0.000 |
| 85107 | 27.819 | 21.005 | 0.000 | 0.000 | 0.000 |
| 85111 | 31.603 | 25.048 | 0.000 | 0.000 | 0.000 |
| 85112 | 55.871 | 40.438 | 0.000 | 0.000 | 0.000 |
| 85116 | 48.097 | 35.034 | 0.000 | 0.000 | 0.000 |
| 85117 | 35.290 | 28.460 | 0.000 | 0.000 | 0.000 |
| 85118 | 34.019 | 27.562 | 0.000 | 0.000 | 0.000 |
| 85119 | 38.074 | 29.916 | 0.000 | 0.000 | 0.000 |
| 85120 | 41.881 | 33.498 | 0.000 | 0.000 | 0.000 |
| 85121 | 36.591 | 27.725 | 0.000 | 0.000 | 0.000 |
| 85122 | 49.568 | 39.759 | 0.000 | 0.000 | 0.000 |
| 85123 | 31.380 | 24.687 | 0.000 | 0.000 | 0.000 |
| 85124 | 40.140 | 30.206 | 0.000 | 0.000 | 0.000 |
| 85125 | 35.205 | 25.923 | 0.000 | 0.000 | 0.000 |
| 85126 | 28.045 | 23.325 | 0.000 | 0.000 | 0.000 |
| 85201 | 38.928 | 30.289 | 0.000 | 0.000 | 0.000 |
| 85202 | 34.068 | 25.790 | 0.000 | 0.000 | 0.000 |
| 85204 | 52.265 | 39.497 | 0.111 | 0.167 | 0.133 |
| 85207 | 37.716 | 30.650 | 0.000 | 0.000 | 0.000 |
| 85208 | 31.322 | 24.411 | 0.000 | 0.000 | 0.000 |
| 85209 | 39.012 | 32.681 | 1.000 | 0.067 | 0.125 |
| 85211 | 32.381 | 24.865 | 0.000 | 0.000 | 0.000 |
| 85214 | 41.436 | 34.427 | 0.000 | 0.000 | 0.000 |
| 85215 | 28.654 | 22.461 | 0.000 | 0.000 | 0.000 |
| 85216 | 36.864 | 29.587 | 0.000 | 0.000 | 0.000 |
| 85217 | 44.747 | 35.363 | 0.000 | 0.000 | 0.000 |
| 85218 | 37.501 | 28.075 | 0.000 | 0.000 | 0.000 |
| 85219 | 35.108 | 26.968 | 0.000 | 0.000 | 0.000 |
| 85220 | 39.571 | 32.664 | 0.000 | 0.000 | 0.000 |
| 85221 | 44.359 | 33.891 | 0.000 | 0.000 | 0.000 |
| 85222 | 45.565 | 36.576 | 0.000 | 0.000 | 0.000 |
| 85224 | 31.016 | 24.017 | 0.000 | 0.000 | 0.000 |
| **Mean** | **37.396** | **29.110** | **0.0435** | **0.0125** | **0.0153** |

---

## 5. Per-Patient Results — PatchTST Soft-DTW

| Patient | RMSE | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 85101 | 25.065 | 19.115 | 0.000 | 0.000 | 0.000 |
| 85102 | 53.483 | 39.103 | 0.038 | 0.125 | 0.059 |
| 85104 | 36.854 | 25.305 | 0.059 | 0.333 | 0.100 |
| 85105 | 44.308 | 32.624 | 0.049 | 0.174 | 0.077 |
| 85106 | 36.400 | 27.567 | 0.000 | 0.000 | 0.000 |
| 85107 | 30.891 | 23.529 | 0.000 | 0.000 | 0.000 |
| 85111 | 35.810 | 28.824 | 0.000 | 0.000 | 0.000 |
| 85112 | 62.497 | 47.419 | 0.000 | 0.000 | 0.000 |
| 85116 | 57.322 | 39.686 | 0.000 | 0.000 | 0.000 |
| 85117 | 38.604 | 29.629 | 0.000 | 0.000 | 0.000 |
| 85118 | 39.245 | 27.166 | 0.219 | 0.500 | 0.304 |
| 85119 | 44.746 | 30.826 | 0.000 | 0.000 | 0.000 |
| 85120 | 47.110 | 37.742 | 0.000 | 0.000 | 0.000 |
| 85121 | 35.688 | 25.956 | 0.048 | 0.200 | 0.077 |
| 85122 | 56.891 | 41.959 | 0.067 | 0.250 | 0.105 |
| 85123 | 35.558 | 28.278 | 0.333 | 0.143 | 0.200 |
| 85124 | 48.036 | 34.817 | 0.034 | 0.250 | 0.061 |
| 85125 | 44.595 | 30.752 | 0.115 | 0.600 | 0.194 |
| 85126 | 32.488 | 25.349 | 0.000 | 0.000 | 0.000 |
| 85201 | 50.152 | 35.888 | 0.000 | 0.000 | 0.000 |
| 85202 | 37.698 | 24.529 | 0.167 | 0.500 | 0.250 |
| 85204 | 64.463 | 46.467 | 0.049 | 0.333 | 0.085 |
| 85207 | 48.606 | 36.006 | 0.500 | 0.200 | 0.286 |
| 85208 | 32.691 | 23.602 | 0.105 | 0.333 | 0.160 |
| 85209 | 41.606 | 32.342 | 0.100 | 0.067 | 0.080 |
| 85211 | 36.250 | 26.818 | 0.500 | 0.500 | 0.500 |
| 85214 | 45.161 | 32.148 | 0.154 | 1.000 | 0.267 |
| 85215 | 32.058 | 23.886 | 0.000 | 0.000 | 0.000 |
| 85216 | 37.173 | 28.007 | 0.143 | 0.091 | 0.111 |
| 85217 | 57.515 | 42.793 | 0.000 | 0.000 | 0.000 |
| 85218 | 43.491 | 32.500 | 0.000 | 0.000 | 0.000 |
| 85219 | 41.943 | 32.181 | 0.000 | 0.000 | 0.000 |
| 85220 | 41.592 | 31.878 | 0.077 | 0.111 | 0.091 |
| 85221 | 52.518 | 37.505 | 0.000 | 0.000 | 0.000 |
| 85222 | 58.539 | 46.653 | 0.000 | 0.000 | 0.000 |
| 85224 | 37.247 | 28.825 | 0.000 | 0.000 | 0.000 |
| **Mean** | **43.453** | **32.158** | **0.0766** | **0.1586** | **0.0835** |

---

## 6. Analysis and Interpretation

### RMSE: best trade-off among alignment losses

LSTM Soft-DTW achieves an RMSE of 37.396, only +3.6% above the baseline LSTM MSE (36.089).
This is by far the smallest RMSE penalty among the three objectives and represents the best
pointwise accuracy–alignment trade-off observed in this study.

The reason Soft-DTW outperforms bounded-lag in RMSE is the nature of the gradient signal.
Bounded-lag selects a single winning shift k* per sample and back-propagates only through that
shift (winner-takes-all). This hard selection can cause the model to progressively learn
time-shifted outputs, which are then penalised at pointwise evaluation. Soft-DTW, by contrast,
distributes gradient across all alignment paths via the soft-minimum, providing a smoother
optimisation landscape that retains more pointwise accuracy while still encouraging temporal
alignment.

PatchTST Soft-DTW (RMSE = 43.453) shows a larger penalty (+10.6%) than LSTM Soft-DTW.
This is likely due to the higher model complexity of PatchTST interacting with the more
complex Soft-DTW loss surface, combined with the O(H²) per-sample cost reducing effective
gradient signal per epoch.

### Hypoglycemia detection: comparable gains to bounded-lag

Both LSTM and PatchTST achieve nearly identical F1 improvements as in the bounded-lag
objective:

- LSTM: F1 0.0054 → 0.0153 (+183%), same as bounded-lag (+185%)
- PatchTST: F1 0.0609 → 0.0835 (+37%), same as bounded-lag (+38%)

This symmetry suggests that the primary driver of event-detection improvement is the shift
from a single-horizon MSE objective to a trajectory-level alignment objective, regardless of
whether that alignment is rigid (bounded-lag) or flexible (DTW).

PatchTST Soft-DTW achieves the highest per-patient F1 values observed in the study: patient
85211 reaches F1 = 0.500 and patient 85214 reaches recall = 1.000 (all events detected).

### LSTM Soft-DTW: overall best balance

Across all six models, LSTM Soft-DTW presents the most balanced profile:
- RMSE only slightly above baseline (+3.6%)
- F1 improvement of ~3× over baseline LSTM
- Computationally efficient (O(H²) per sample, but H=12 is small)

### Key trade-off

Soft-DTW encodes temporal alignment as a soft, differentiable constraint. The resulting models
learn to predict the shape and direction of glucose dynamics rather than purely optimising for
the exact value at t+60 min. For hypoglycemia prevention — where detecting a descent trend
early is more valuable than exact magnitude — this is a clinically motivated objective.
The γ = 1.0 parameter controls the softness of alignment; smaller γ would approach hard DTW
alignment (sharper gradients, risk of vanishing) while larger γ would smooth out differences.
