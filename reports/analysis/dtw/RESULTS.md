# Objective 1 (DTW variant): Soft-DTW Loss

**Branch:** `devBranch-offset-loss-dtw`
**Loss function:** Soft-DTW (gamma = 1.0)
**Models:** LSTM and PatchTST with baseline hyperparameters (same as Objective 1 bounded-lag)

---

## 1. Loss Function

Soft-DTW is a differentiable relaxation of Dynamic Time Warping [4]. Instead of the hard
minimum in the DTW recursion, Soft-DTW uses a soft-minimum via the log-sum-exp operator
controlled by smoothing parameter gamma:

```
soft-min_gamma(a, b, c) = -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))
```

As gamma approaches 0, Soft-DTW recovers standard DTW. For gamma = 1.0 the loss provides
a smooth gradient signal throughout training. Unlike bounded-lag MSE, which assumes a
global constant shift, Soft-DTW allows non-linear, sample-wise temporal warping, permitting
the model to align predicted and true trajectories more flexibly.

The computational cost is O(H^2) per sample, where H = 12 is the prediction horizon.
Evaluation remains strict pointwise RMSE at h = 11 (60 min), preserving a consistent
train/eval mismatch design across all three objectives.

Soft-DTW was included as an optional variant of Objective 1, as described in the proposal
[3]. It is included because it was stable and computationally feasible at this horizon.

---

## 2. Hyperparameters (Baseline Optuna, patient 85202)

The same hyperparameters as the baseline are used for both architectures. No separate
Optuna tuning was performed for the DTW variant. This ensures the comparison isolates the
effect of the loss function.

| Parameter | LSTM | PatchTST |
|---|---|---|
| hidden_size / d_model | 256 | 64 |
| num_layers / n_layers | 2 | 3 |
| dropout | 0.159 | 0.025 |
| learning rate | 9.61e-4 | 1.21e-4 |
| batch_size | 128 | 128 |
| dim_ff | — | 256 |
| n_heads | — | 4 |

Note: an earlier version of this script used Optuna-tuned hyperparameters specific to
the DTW loss, which found d_model=256 for PatchTST (4x larger than baseline). That
comparison was unfair and produced an inflated F1 gain (+37%) for PatchTST. The current
results use the shared baseline HP, reducing the apparent PatchTST gain to +2.5%.

---

## 3. Summary Results vs Baseline

Event metrics at tau = 3 steps (15 min), threshold = 70 mg/dL. Delta* measured
within +-12 steps (+-60 min) by minimising RMSE after shift correction.

| Model | RMSE | lag_rmse | Delta* mean (min) | Delta* median (min) | Delta* std (min) | MAE | F1 | ΔF1 vs baseline |
|---|---|---|---|---|---|---|---|---|
| LSTM Baseline | 36.089 | 21.649 | -51.2 | -52.5 | 9.1 | 27.118 | 0.0054 | — |
| **LSTM Soft-DTW** | **37.232** | **28.668** | **-45.4** | **-45.0** | **13.4** | **28.760** | **0.0137** | **+154%** |
| PatchTST Baseline | 39.271 | 14.213 | -55.3 | -55.0 | 3.1 | 29.001 | 0.0609 | — |
| **PatchTST Soft-DTW** | **43.235** | **24.006** | **-54.4** | **-55.0** | **3.1** | **32.251** | **0.0624** | **+2.5%** |

---

## 4. Per-Patient Results — LSTM Soft-DTW

| Patient | RMSE | lag_rmse | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 85101 | 21.273 | 15.238 | 16.539 | 0.000 | 0.000 | 0.000 |
| 85102 | 46.579 | 36.826 | 37.596 | 0.000 | 0.000 | 0.000 |
| 85104 | 26.942 | 22.628 | 20.466 | 0.000 | 0.000 | 0.000 |
| 85105 | 33.328 | 32.797 | 21.258 | 0.400 | 0.174 | 0.242 |
| 85106 | 32.590 | 27.474 | 23.800 | 0.000 | 0.000 | 0.000 |
| 85107 | 27.084 | 16.591 | 20.146 | 0.000 | 0.000 | 0.000 |
| 85111 | 30.996 | 18.670 | 24.420 | 0.000 | 0.000 | 0.000 |
| 85112 | 51.708 | 44.877 | 38.630 | 0.000 | 0.000 | 0.000 |
| 85116 | 48.685 | 30.275 | 35.995 | 0.000 | 0.000 | 0.000 |
| 85117 | 37.178 | 37.092 | 30.160 | 0.000 | 0.000 | 0.000 |
| 85118 | 31.683 | 24.249 | 25.120 | 0.000 | 0.000 | 0.000 |
| 85119 | 39.098 | 31.154 | 31.087 | 0.000 | 0.000 | 0.000 |
| 85120 | 42.554 | 32.180 | 33.228 | 0.000 | 0.000 | 0.000 |
| 85121 | 34.880 | 29.740 | 25.643 | 0.000 | 0.000 | 0.000 |
| 85122 | 50.227 | 46.622 | 40.122 | 0.000 | 0.000 | 0.000 |
| 85123 | 32.104 | 17.964 | 24.780 | 1.000 | 0.143 | 0.250 |
| 85124 | 40.815 | 31.222 | 30.765 | 0.000 | 0.000 | 0.000 |
| 85125 | 35.881 | 24.838 | 26.251 | 0.000 | 0.000 | 0.000 |
| 85126 | 26.074 | 18.710 | 20.678 | 0.000 | 0.000 | 0.000 |
| 85201 | 40.557 | 34.381 | 30.575 | 0.000 | 0.000 | 0.000 |
| 85202 | 34.217 | 21.678 | 25.767 | 0.000 | 0.000 | 0.000 |
| 85204 | 52.259 | 34.793 | 39.860 | 0.000 | 0.000 | 0.000 |
| 85207 | 38.197 | 34.038 | 30.619 | 0.000 | 0.000 | 0.000 |
| 85208 | 30.224 | 25.702 | 23.342 | 0.000 | 0.000 | 0.000 |
| 85209 | 37.202 | 29.432 | 30.803 | 0.000 | 0.000 | 0.000 |
| 85211 | 33.625 | 21.089 | 25.820 | 0.000 | 0.000 | 0.000 |
| 85214 | 41.683 | 34.260 | 34.735 | 0.000 | 0.000 | 0.000 |
| 85215 | 29.018 | 23.540 | 22.692 | 0.000 | 0.000 | 0.000 |
| 85216 | 35.505 | 27.805 | 28.450 | 0.000 | 0.000 | 0.000 |
| 85217 | 45.278 | 31.523 | 36.299 | 0.000 | 0.000 | 0.000 |
| 85218 | 38.504 | 25.245 | 28.538 | 0.000 | 0.000 | 0.000 |
| 85219 | 33.845 | 25.997 | 25.156 | 0.000 | 0.000 | 0.000 |
| 85220 | 37.707 | 29.652 | 30.353 | 0.000 | 0.000 | 0.000 |
| 85221 | 45.053 | 31.600 | 33.839 | 0.000 | 0.000 | 0.000 |
| 85222 | 46.050 | 31.340 | 37.183 | 0.000 | 0.000 | 0.000 |
| 85224 | 31.757 | 30.822 | 24.633 | 0.000 | 0.000 | 0.000 |
| **Mean** | **37.232** | **28.668** | **28.760** | **0.0389** | **0.0088** | **0.0137** |

---

## 5. Per-Patient Results — PatchTST Soft-DTW

| Patient | RMSE | lag_rmse | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 85101 | 22.839 | 10.783 | 16.995 | 0.000 | 0.000 | 0.000 |
| 85102 | 56.063 | 30.799 | 43.297 | 0.000 | 0.000 | 0.000 |
| 85104 | 34.449 | 22.345 | 24.810 | 0.000 | 0.000 | 0.000 |
| 85105 | 46.411 | 38.314 | 34.888 | 0.036 | 0.043 | 0.039 |
| 85106 | 37.784 | 26.771 | 29.171 | 0.000 | 0.000 | 0.000 |
| 85107 | 32.356 | 15.627 | 23.426 | 0.083 | 0.200 | 0.118 |
| 85111 | 36.529 | 19.701 | 27.965 | 0.125 | 1.000 | 0.222 |
| 85112 | 59.771 | 32.632 | 45.580 | 0.000 | 0.000 | 0.000 |
| 85116 | 55.274 | 29.174 | 38.504 | 0.000 | 0.000 | 0.000 |
| 85117 | 42.939 | 28.902 | 32.268 | 0.040 | 0.111 | 0.059 |
| 85118 | 35.159 | 17.527 | 25.495 | 0.000 | 0.000 | 0.000 |
| 85119 | 44.772 | 22.075 | 30.703 | 0.000 | 0.000 | 0.000 |
| 85120 | 45.304 | 25.089 | 35.980 | 0.000 | 0.000 | 0.000 |
| 85121 | 32.590 | 15.584 | 23.420 | 0.000 | 0.000 | 0.000 |
| 85122 | 57.263 | 40.548 | 43.747 | 0.091 | 0.250 | 0.133 |
| 85123 | 36.923 | 20.240 | 29.363 | 0.000 | 0.000 | 0.000 |
| 85124 | 46.742 | 23.596 | 34.353 | 0.000 | 0.000 | 0.000 |
| 85125 | 42.088 | 23.065 | 29.546 | 0.107 | 0.600 | 0.182 |
| 85126 | 30.335 | 14.874 | 20.585 | 0.000 | 0.000 | 0.000 |
| 85201 | 52.958 | 28.620 | 41.367 | 0.000 | 0.000 | 0.000 |
| 85202 | 40.554 | 19.955 | 30.330 | 0.000 | 0.000 | 0.000 |
| 85204 | 62.758 | 33.474 | 45.411 | 0.056 | 0.333 | 0.095 |
| 85207 | 47.222 | 29.603 | 35.431 | 0.667 | 0.200 | 0.308 |
| 85208 | 34.208 | 15.213 | 24.732 | 0.091 | 0.333 | 0.143 |
| 85209 | 40.658 | 21.690 | 30.387 | 0.086 | 0.333 | 0.137 |
| 85211 | 37.185 | 19.193 | 27.272 | 0.400 | 0.500 | 0.444 |
| 85214 | 46.141 | 22.290 | 35.192 | 0.000 | 0.000 | 0.000 |
| 85215 | 33.570 | 18.626 | 26.303 | 0.000 | 0.000 | 0.000 |
| 85216 | 37.921 | 18.610 | 28.583 | 0.000 | 0.000 | 0.000 |
| 85217 | 55.179 | 28.131 | 40.948 | 0.000 | 0.000 | 0.000 |
| 85218 | 44.484 | 22.797 | 33.917 | 0.065 | 0.200 | 0.098 |
| 85219 | 38.044 | 19.932 | 26.954 | 0.000 | 0.000 | 0.000 |
| 85220 | 41.473 | 24.988 | 31.323 | 0.143 | 0.333 | 0.200 |
| 85221 | 53.319 | 28.490 | 39.822 | 0.000 | 0.000 | 0.000 |
| 85222 | 58.033 | 30.445 | 46.351 | 0.000 | 0.000 | 0.000 |
| 85224 | 37.165 | 24.524 | 26.627 | 0.042 | 0.200 | 0.069 |
| **Mean** | **43.235** | **24.006** | **32.251** | **0.0564** | **0.1288** | **0.0624** |

---

## 6. Time-Shift Analysis (Delta*)

Delta* is the correction lag in minutes that minimises RMSE(y_true, shift(y_pred, k))
within a search window of +-12 steps (+-60 min). A negative Delta* means the prediction
must be advanced to achieve the best alignment, i.e., the original prediction is that
many minutes late.

| Model | Delta* mean (min) | Delta* median (min) | Delta* std (min) |
|---|---|---|---|
| LSTM Baseline | -51.2 | -52.5 | 9.1 |
| LSTM Soft-DTW | -45.4 | -45.0 | 13.4 |
| PatchTST Baseline | -55.3 | -55.0 | 3.1 |
| PatchTST Soft-DTW | -54.4 | -55.0 | 3.1 |

LSTM Soft-DTW shows a reduction in mean Delta* from -51.2 to -45.4 min (improvement of
5.8 min) but with increased variability (std 13.4 vs 9.1 min). This means Soft-DTW shifts
some predictions earlier but introduces more patient-to-patient variation in the offset.
PatchTST shows no change. Neither improvement is clinically meaningful.

---

## 7. Analysis and Interpretation

### RMSE: small penalty for LSTM, larger for PatchTST

LSTM Soft-DTW achieves RMSE = 37.232, only +3.2% above baseline LSTM (36.089). PatchTST
Soft-DTW (RMSE = 43.235) shows a larger penalty (+10.1%). Soft-DTW distributes the gradient
signal across all alignment paths rather than selecting a single winning shift, which appears
to better preserve pointwise accuracy for LSTM but not for PatchTST.

### lag_rmse increases: time-shift artefact not reduced

Despite optimising for temporal alignment, Soft-DTW does not reduce lag_rmse:

- LSTM: 21.649 to 28.668 (+32%)
- PatchTST: 14.213 to 24.006 (+69%)

Soft-DTW is by design invariant to temporal distortions: the loss is minimised when the
predicted trajectory can be warped onto the true trajectory within the full alignment space.
A model that consistently predicts glucose dynamics too late can still achieve low Soft-DTW
loss by warping the alignment path to match. The model never receives a gradient signal that
penalises the direction of time offset. As a result, predictions become more varied in their
per-sample offset, making a single global constant shift (lag_rmse) a worse fit.

### Hypoglycemia detection: modest improvement for LSTM, negligible for PatchTST

With controlled baseline hyperparameters:

- LSTM: F1 0.0054 to 0.0137 (+154%)
- PatchTST: F1 0.0609 to 0.0624 (+2.5%)

The LSTM improvement is real but small in absolute terms. The PatchTST improvement is
negligible. This is a correction from an earlier version: a run with DTW-specific Optuna
HP (d_model=256 vs baseline 64) produced PatchTST F1=0.0835 (+37%). After fixing the
HP to match the baseline, that gain disappears. The earlier result was driven by larger
model capacity, not by the DTW loss.

### Soft-DTW does not address the root cause of time-shift

Soft-DTW is appropriate for tasks where trajectory shape matters but timing does not, such
as gesture recognition or sequence similarity search [4]. For CGM glucose forecasting, the
goal is to predict the value at exactly t+60 min. Temporal invariance is a liability here
because it allows the model to produce delayed predictions that match the trajectory shape
but at the wrong time. This is confirmed by the unchanged or worsened Delta* and lag_rmse
across both architectures.

---

## References

[3] Pattern Recognition Group, University of Bern. Glucose Prediction Proposal.
    Internal unpublished manuscript.

[4] H. Sakoe and S. Chiba. Dynamic programming algorithm optimization for spoken word
    recognition. IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 26,
    pp. 43-49, 1978.
