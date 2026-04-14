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

| Model | RMSE | lag_rmse | Δ RMSE vs baseline | MAE | Precision | Recall | F1 | ΔF1 vs baseline |
|---|---|---|---|---|---|---|---|---|
| LSTM MSE (baseline) | 36.089 | 21.649 | — | 27.118 | 0.0104 | 0.0036 | 0.0054 | — |
| **LSTM Soft-DTW** | **37.232** | **28.668** | **+3.2%** | **28.760** | 0.0389 | 0.0088 | 0.0137 | +154% |
| PatchTST MSE (baseline) | 39.271 | 14.213 | — | 29.001 | 0.0482 | 0.0888 | 0.0609 | — |
| **PatchTST Soft-DTW** | **43.235** | **24.006** | **+10.1%** | **32.251** | 0.0564 | 0.1288 | 0.0624 | +2.5% |

---

## 4. Per-Patient Results — LSTM Soft-DTW

| Patient | RMSE | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 85101 | 21.273 | 16.539 | 0.000 | 0.000 | 0.000 |
| 85102 | 46.579 | 37.596 | 0.000 | 0.000 | 0.000 |
| 85104 | 26.942 | 20.466 | 0.000 | 0.000 | 0.000 |
| 85105 | 33.328 | 21.258 | 0.400 | 0.174 | 0.242 |
| 85106 | 32.590 | 23.800 | 0.000 | 0.000 | 0.000 |
| 85107 | 27.084 | 20.146 | 0.000 | 0.000 | 0.000 |
| 85111 | 30.996 | 24.420 | 0.000 | 0.000 | 0.000 |
| 85112 | 51.708 | 38.630 | 0.000 | 0.000 | 0.000 |
| 85116 | 48.685 | 35.995 | 0.000 | 0.000 | 0.000 |
| 85117 | 37.178 | 30.160 | 0.000 | 0.000 | 0.000 |
| 85118 | 31.683 | 25.120 | 0.000 | 0.000 | 0.000 |
| 85119 | 39.098 | 31.087 | 0.000 | 0.000 | 0.000 |
| 85120 | 42.554 | 33.228 | 0.000 | 0.000 | 0.000 |
| 85121 | 34.880 | 25.643 | 0.000 | 0.000 | 0.000 |
| 85122 | 50.227 | 40.122 | 0.000 | 0.000 | 0.000 |
| 85123 | 32.104 | 24.780 | 1.000 | 0.143 | 0.250 |
| 85124 | 40.815 | 30.765 | 0.000 | 0.000 | 0.000 |
| 85125 | 35.881 | 26.251 | 0.000 | 0.000 | 0.000 |
| 85126 | 26.074 | 20.678 | 0.000 | 0.000 | 0.000 |
| 85201 | 40.557 | 30.575 | 0.000 | 0.000 | 0.000 |
| 85202 | 34.217 | 25.767 | 0.000 | 0.000 | 0.000 |
| 85204 | 52.259 | 39.860 | 0.000 | 0.000 | 0.000 |
| 85207 | 38.197 | 30.619 | 0.000 | 0.000 | 0.000 |
| 85208 | 30.224 | 23.342 | 0.000 | 0.000 | 0.000 |
| 85209 | 37.202 | 30.803 | 0.000 | 0.000 | 0.000 |
| 85211 | 33.625 | 25.820 | 0.000 | 0.000 | 0.000 |
| 85214 | 41.683 | 34.735 | 0.000 | 0.000 | 0.000 |
| 85215 | 29.018 | 22.692 | 0.000 | 0.000 | 0.000 |
| 85216 | 35.505 | 28.450 | 0.000 | 0.000 | 0.000 |
| 85217 | 45.278 | 36.299 | 0.000 | 0.000 | 0.000 |
| 85218 | 38.504 | 28.538 | 0.000 | 0.000 | 0.000 |
| 85219 | 33.845 | 25.156 | 0.000 | 0.000 | 0.000 |
| 85220 | 37.707 | 30.353 | 0.000 | 0.000 | 0.000 |
| 85221 | 45.053 | 33.839 | 0.000 | 0.000 | 0.000 |
| 85222 | 46.050 | 37.183 | 0.000 | 0.000 | 0.000 |
| 85224 | 31.757 | 24.633 | 0.000 | 0.000 | 0.000 |
| **Mean** | **37.232** | **28.760** | **0.0389** | **0.0088** | **0.0137** |

---

## 5. Per-Patient Results — PatchTST Soft-DTW

| Patient | RMSE | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 85101 | 22.839 | 16.995 | 0.000 | 0.000 | 0.000 |
| 85102 | 56.063 | 43.297 | 0.000 | 0.000 | 0.000 |
| 85104 | 34.449 | 24.810 | 0.000 | 0.000 | 0.000 |
| 85105 | 46.411 | 34.888 | 0.036 | 0.043 | 0.039 |
| 85106 | 37.784 | 29.171 | 0.000 | 0.000 | 0.000 |
| 85107 | 32.356 | 23.426 | 0.083 | 0.200 | 0.118 |
| 85111 | 36.529 | 27.965 | 0.125 | 1.000 | 0.222 |
| 85112 | 59.771 | 45.580 | 0.000 | 0.000 | 0.000 |
| 85116 | 55.274 | 38.504 | 0.000 | 0.000 | 0.000 |
| 85117 | 42.939 | 32.268 | 0.040 | 0.111 | 0.059 |
| 85118 | 35.159 | 25.495 | 0.000 | 0.000 | 0.000 |
| 85119 | 44.772 | 30.703 | 0.000 | 0.000 | 0.000 |
| 85120 | 45.304 | 35.980 | 0.000 | 0.000 | 0.000 |
| 85121 | 32.590 | 23.420 | 0.000 | 0.000 | 0.000 |
| 85122 | 57.263 | 43.747 | 0.091 | 0.250 | 0.133 |
| 85123 | 36.923 | 29.363 | 0.000 | 0.000 | 0.000 |
| 85124 | 46.742 | 34.353 | 0.000 | 0.000 | 0.000 |
| 85125 | 42.088 | 29.546 | 0.107 | 0.600 | 0.182 |
| 85126 | 30.335 | 20.585 | 0.000 | 0.000 | 0.000 |
| 85201 | 52.958 | 41.367 | 0.000 | 0.000 | 0.000 |
| 85202 | 40.554 | 30.330 | 0.000 | 0.000 | 0.000 |
| 85204 | 62.758 | 45.411 | 0.056 | 0.333 | 0.095 |
| 85207 | 47.223 | 35.431 | 0.667 | 0.200 | 0.308 |
| 85208 | 34.208 | 24.732 | 0.091 | 0.333 | 0.143 |
| 85209 | 40.658 | 30.387 | 0.086 | 0.333 | 0.137 |
| 85211 | 37.185 | 27.272 | 0.400 | 0.500 | 0.444 |
| 85214 | 46.141 | 35.192 | 0.000 | 0.000 | 0.000 |
| 85215 | 33.570 | 26.303 | 0.000 | 0.000 | 0.000 |
| 85216 | 37.921 | 28.583 | 0.000 | 0.000 | 0.000 |
| 85217 | 55.179 | 40.948 | 0.000 | 0.000 | 0.000 |
| 85218 | 44.484 | 33.917 | 0.065 | 0.200 | 0.098 |
| 85219 | 38.044 | 26.954 | 0.000 | 0.000 | 0.000 |
| 85220 | 41.473 | 31.323 | 0.143 | 0.333 | 0.200 |
| 85221 | 53.319 | 39.822 | 0.000 | 0.000 | 0.000 |
| 85222 | 58.033 | 46.351 | 0.000 | 0.000 | 0.000 |
| 85224 | 37.165 | 26.627 | 0.042 | 0.200 | 0.069 |
| **Mean** | **43.235** | **32.251** | **0.0564** | **0.1288** | **0.0624** |

---

## 6. Analysis and Interpretation

### RMSE: small penalty, consistent across HP

LSTM Soft-DTW achieves RMSE = 37.232, only +3.2% above baseline LSTM MSE (36.089). This is
the smallest RMSE penalty across the three objectives and is consistent with the bounded-lag
result (+3.0%). Soft-DTW distributes the gradient signal across all alignment paths via the
soft-minimum rather than selecting a single winning shift, which appears to better preserve
pointwise accuracy.

PatchTST Soft-DTW (RMSE = 43.235) shows a larger penalty (+10.1%) compared to baseline
PatchTST (39.271). This is consistent with the bounded-lag result for PatchTST (+0.1%) being
an exception — the Soft-DTW loss surface is more complex and PatchTST appears to trade
pointwise accuracy for trajectory-level alignment more than LSTM does.

### lag_rmse increases — time-shift artefact not reduced

Despite optimising for temporal alignment, Soft-DTW does not reduce lag_rmse compared to
baseline:

- LSTM: 21.649 → 28.668 (+32%)
- PatchTST: 14.213 → 24.006 (+69%)

This mirrors the bounded-lag finding. Soft-DTW is by design invariant to temporal distortions:
the loss is minimised when the predicted trajectory can be warped onto the true trajectory
within the full alignment space. A model that consistently predicts glucose dynamics 15–30
minutes too late can still achieve low Soft-DTW loss by warping the alignment path accordingly.
The model never receives a gradient signal that specifically penalises the direction of time
offset. As a result, predictions become more varied in their per-sample offset, making a single
global constant shift (lag_rmse) a worse fit.

### Hypoglycemia detection: modest improvement for LSTM, negligible for PatchTST

With corrected baseline hyperparameters, the F1 improvements are substantially smaller than
preliminary results suggested:

- LSTM: F1 0.0054 → 0.0137 (+154%)
- PatchTST: F1 0.0609 → 0.0624 (+2.5%)

The LSTM improvement is real but small in absolute terms (only two patients detect events in
both runs). The PatchTST improvement is negligible. This is a critical correction from
earlier results: a previous run with DTW-specific Optuna hyperparameters (d_model=256 vs
baseline d_model=64) produced F1=0.0835 for PatchTST Soft-DTW (+37%). After controlling for
architecture by using the shared baseline HP, that apparent gain disappears almost entirely.
The earlier improvement was driven by the larger model capacity, not by the DTW loss itself.

This finding demonstrates that the ablation design — fixing hyperparameters across objectives —
is essential for isolating the effect of the loss function. Without it, a confound (model size)
masked the true contribution of Soft-DTW.

### LSTM Soft-DTW: most balanced profile among alignment losses

LSTM Soft-DTW remains the best-balanced variant:
- Smallest RMSE penalty (+3.2%)
- Largest absolute F1 gain among alignment objectives for LSTM (+154%)
- Computationally tractable (O(H²) per sample, H=12)

### Key limitation of Soft-DTW for this task

Soft-DTW is designed for applications where exact timing does not matter — recognising the
shape of a glucose excursion regardless of when it peaks. For the specific clinical goal of
predicting glucose at exactly t+60 min, this invariance is a liability. The model learns good
curve shapes but continues to produce them at shifted time points. Combined with the loss of
the architecture confound, Soft-DTW does not provide a meaningful advantage over MSE for
this forecasting task when hyperparameters are properly controlled.
