# Objective 2: Multi-Step MSE Training

**Branch:** `devBranch-multi-step`
**Training objective:** MSE over all 12 horizon steps (direct multi-horizon forecasting)
**Models:** LSTM and PatchTST with baseline hyperparameters

---

## 1. Approach

The baseline trains on a single output at h = 11 (60-min horizon). Multi-step training
instead computes MSE over all 12 steps of the predicted trajectory simultaneously.
The hypothesis is that supervising every intermediate step forces the model to track
the glucose trajectory more closely at each time point, reducing the tendency to produce
predictions that are systematically delayed relative to the true trajectory [3].

The evaluation remains at h = 11 (60 min) only, so the comparison with the baseline is
fair: both models are assessed on the same output step.

The same backbone architecture and baseline hyperparameters from Optuna tuning on patient
85202 are used for both LSTM and PatchTST. Only the training objective changes.

---

## 2. Hyperparameters (Baseline Optuna, patient 85202)

| Parameter | LSTM | PatchTST |
|---|---|---|
| hidden_size / d_model | 256 | 64 |
| num_layers / n_layers | 2 | 3 |
| dropout | 0.159 | 0.025 |
| learning rate | 9.61e-4 | 1.21e-4 |
| batch_size | 128 | 128 |
| dim_ff | — | 256 |
| n_heads | — | 4 |

---

## 3. Summary Results vs Baseline

Event metrics at tau = 3 steps (15 min), threshold = 70 mg/dL. Δ* measured
within ±12 steps (±60 min) by minimising RMSE after shift correction.

| Model | RMSE | lag_rmse | Δ* mean (min) | Δ* median (min) | Δ* std (min) | MAE | F1 | F2 |
|---|---|---|---|---|---|---|---|---|
| LSTM Baseline | 36.089 | 21.649 | -51.2 | -52.5 | 9.1 | 27.118 | 0.0054 | 0.0042 |
| **LSTM Multi-Step** | **35.586** | **20.744** | **-50.1** | **-50.0** | **8.6** | **26.803** | **0.0098** | **0.0095** |
| PatchTST Baseline | 39.271 | 14.213 | -55.3 | -55.0 | 3.1 | 29.001 | 0.0609 | 0.0742 |
| **PatchTST Multi-Step** | **39.051** | **14.528** | **-54.6** | **-55.0** | **3.0** | **28.837** | **0.0716** | **0.0851** |

---

## 4. Per-Patient Results — LSTM Multi-Step

| Patient | RMSE | lag_rmse | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 85101 | 20.163 | 8.308 | 14.911 | 0.000 | 0.000 | 0.000 |
| 85102 | 44.354 | 29.302 | 35.198 | 0.000 | 0.000 | 0.000 |
| 85104 | 26.087 | 16.626 | 19.125 | 0.000 | 0.000 | 0.000 |
| 85105 | 33.175 | 32.810 | 21.071 | 0.267 | 0.174 | 0.211 |
| 85106 | 29.995 | 20.501 | 22.008 | 0.000 | 0.000 | 0.000 |
| 85107 | 26.808 | 14.305 | 19.786 | 0.000 | 0.000 | 0.000 |
| 85111 | 31.425 | 15.301 | 24.526 | 0.000 | 0.000 | 0.000 |
| 85112 | 47.673 | 33.998 | 35.772 | 0.000 | 0.000 | 0.000 |
| 85116 | 47.305 | 18.046 | 33.562 | 0.000 | 0.000 | 0.000 |
| 85117 | 32.586 | 26.309 | 25.567 | 0.000 | 0.000 | 0.000 |
| 85118 | 30.853 | 16.021 | 23.143 | 0.000 | 0.000 | 0.000 |
| 85119 | 38.078 | 19.089 | 28.469 | 0.000 | 0.000 | 0.000 |
| 85120 | 38.385 | 18.383 | 30.263 | 0.000 | 0.000 | 0.000 |
| 85121 | 31.766 | 25.977 | 22.930 | 0.000 | 0.000 | 0.000 |
| 85122 | 45.928 | 32.264 | 35.643 | 0.000 | 0.000 | 0.000 |
| 85123 | 31.002 | 14.316 | 23.947 | 0.000 | 0.000 | 0.000 |
| 85124 | 38.881 | 22.023 | 28.245 | 0.000 | 0.000 | 0.000 |
| 85125 | 34.717 | 17.452 | 25.057 | 0.000 | 0.000 | 0.000 |
| 85126 | 25.865 | 13.930 | 19.924 | 0.000 | 0.000 | 0.000 |
| 85201 | 39.274 | 24.699 | 29.549 | 0.000 | 0.000 | 0.000 |
| 85202 | 33.576 | 16.611 | 24.241 | 0.000 | 0.000 | 0.000 |
| 85204 | 52.364 | 30.203 | 37.712 | 0.125 | 0.167 | 0.143 |
| 85207 | 35.674 | 22.131 | 28.334 | 0.000 | 0.000 | 0.000 |
| 85208 | 27.995 | 18.154 | 21.525 | 0.000 | 0.000 | 0.000 |
| 85209 | 35.238 | 19.220 | 27.864 | 0.000 | 0.000 | 0.000 |
| 85211 | 32.654 | 15.533 | 24.885 | 0.000 | 0.000 | 0.000 |
| 85214 | 37.600 | 22.059 | 29.194 | 0.000 | 0.000 | 0.000 |
| 85215 | 26.970 | 15.502 | 20.623 | 0.000 | 0.000 | 0.000 |
| 85216 | 33.677 | 22.342 | 26.414 | 0.000 | 0.000 | 0.000 |
| 85217 | 44.102 | 24.963 | 34.381 | 0.000 | 0.000 | 0.000 |
| 85218 | 37.604 | 15.518 | 27.896 | 0.000 | 0.000 | 0.000 |
| 85219 | 31.900 | 18.350 | 22.706 | 0.000 | 0.000 | 0.000 |
| 85220 | 37.710 | 24.018 | 30.152 | 0.000 | 0.000 | 0.000 |
| 85221 | 44.358 | 21.889 | 32.090 | 0.000 | 0.000 | 0.000 |
| 85222 | 45.495 | 18.097 | 35.723 | 0.000 | 0.000 | 0.000 |
| 85224 | 29.859 | 22.526 | 22.483 | 0.000 | 0.000 | 0.000 |
| **Mean** | **35.586** | **20.744** | **26.803** | **0.0109** | **0.0095** | **0.0098** |

---

## 5. Per-Patient Results — PatchTST Multi-Step

| Patient | RMSE | lag_rmse | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 85101 | 21.299 | 7.120 | 15.853 | 0.000 | 0.000 | 0.000 |
| 85102 | 49.167 | 18.555 | 36.860 | 0.000 | 0.000 | 0.000 |
| 85104 | 32.430 | 15.516 | 22.919 | 0.000 | 0.000 | 0.000 |
| 85105 | 41.313 | 24.761 | 30.644 | 0.044 | 0.087 | 0.059 |
| 85106 | 34.460 | 13.412 | 25.389 | 0.125 | 0.167 | 0.143 |
| 85107 | 28.810 | 11.434 | 21.415 | 0.000 | 0.000 | 0.000 |
| 85111 | 33.493 | 10.323 | 26.523 | 0.000 | 0.000 | 0.000 |
| 85112 | 52.365 | 17.837 | 38.819 | 0.000 | 0.000 | 0.000 |
| 85116 | 50.095 | 15.932 | 35.313 | 0.000 | 0.000 | 0.000 |
| 85117 | 36.066 | 17.778 | 27.593 | 0.000 | 0.000 | 0.000 |
| 85118 | 33.660 | 9.969 | 23.251 | 0.033 | 0.071 | 0.045 |
| 85119 | 41.215 | 15.924 | 29.817 | 0.000 | 0.000 | 0.000 |
| 85120 | 41.165 | 15.483 | 31.846 | 0.000 | 0.000 | 0.000 |
| 85121 | 30.372 | 12.150 | 22.076 | 0.000 | 0.000 | 0.000 |
| 85122 | 52.242 | 24.629 | 39.287 | 0.000 | 0.000 | 0.000 |
| 85123 | 31.696 | 9.879 | 23.928 | 0.167 | 0.286 | 0.211 |
| 85124 | 42.221 | 14.892 | 30.866 | 0.000 | 0.000 | 0.000 |
| 85125 | 39.275 | 14.663 | 26.899 | 0.000 | 0.000 | 0.000 |
| 85126 | 28.859 | 9.516 | 20.505 | 0.000 | 0.000 | 0.000 |
| 85201 | 46.742 | 20.182 | 35.829 | 0.000 | 0.000 | 0.000 |
| 85202 | 36.777 | 9.095 | 24.979 | 0.250 | 0.500 | 0.333 |
| 85204 | 59.059 | 25.274 | 42.779 | 0.083 | 0.333 | 0.133 |
| 85207 | 39.423 | 16.054 | 30.177 | 0.167 | 0.200 | 0.182 |
| 85208 | 30.695 | 8.497 | 23.323 | 0.000 | 0.000 | 0.000 |
| 85209 | 36.296 | 12.187 | 27.412 | 0.053 | 0.133 | 0.075 |
| 85211 | 35.610 | 14.184 | 26.649 | 0.333 | 0.500 | 0.400 |
| 85214 | 41.635 | 16.349 | 30.801 | 0.500 | 0.500 | 0.500 |
| 85215 | 30.504 | 10.627 | 23.145 | 0.000 | 0.000 | 0.000 |
| 85216 | 34.961 | 10.774 | 25.514 | 0.000 | 0.000 | 0.000 |
| 85217 | 48.895 | 15.467 | 36.118 | 0.100 | 0.500 | 0.167 |
| 85218 | 39.751 | 10.266 | 29.660 | 0.133 | 0.200 | 0.160 |
| 85219 | 35.747 | 12.087 | 25.946 | 0.000 | 0.000 | 0.000 |
| 85220 | 38.994 | 16.237 | 29.293 | 0.125 | 0.111 | 0.118 |
| 85221 | 47.303 | 15.339 | 33.845 | 0.000 | 0.000 | 0.000 |
| 85222 | 49.866 | 15.058 | 38.561 | 0.032 | 0.125 | 0.051 |
| 85224 | 33.389 | 15.567 | 24.310 | 0.000 | 0.000 | 0.000 |
| **Mean** | **39.051** | **14.528** | **28.837** | **0.0596** | **0.1032** | **0.0716** |

---

## 6. Time-Shift Analysis (Δ*)

Δ* is the correction lag in minutes that minimises RMSE(y_true, shift(y_pred, k))
within a search window of ±12 steps (±60 min). A negative Δ* means the prediction
must be advanced to achieve the best alignment, i.e., the original prediction is
that many minutes late.

For a 60-minute forecast horizon, Δ* ≈ -55 min means the model effectively predicts
only 60 - 55 = 5 minutes into the future rather than 60 minutes. This is the central
time-shift artefact identified in the proposal [3].

| Model | Δ* mean (min) | Δ* median (min) | Δ* std (min) |
|---|---|---|---|
| LSTM Baseline | -51.2 | -52.5 | 9.1 |
| LSTM Multi-Step | -50.1 | -50.0 | 8.6 |
| PatchTST Baseline | -55.3 | -55.0 | 3.1 |
| PatchTST Multi-Step | -54.6 | -55.0 | 3.0 |

All models show a strong systematic time-shift. A 60-minute forecast effectively
predicts approximately 5 to 10 minutes into the future. Multi-step training reduces
Δ* by approximately 1 minute for LSTM and less than 1 minute for PatchTST. Neither
improvement is clinically meaningful.

---

## 7. Analysis

### LSTM Multi-Step: small RMSE improvement, no shift reduction

LSTM Multi-Step reduces RMSE by 0.5 (35.586 vs 36.089) and MAE by 0.3 (26.803 vs
27.118). These are minor improvements. The lag_rmse is almost unchanged (20.744 vs
21.649), and Δ* barely changes (-50.1 vs -51.2 min). Event detection improves from
F1=0.0054 to F1=0.0098 at tau=15 min, a relative gain of +81%, but the absolute value
remains very low.

Multi-step supervision gives the LSTM model slightly more training signal (12 MSE
terms instead of 1), which reduces overall prediction error. However, it does not
address the cause of the time-shift: the model can still minimise the multi-step loss
by producing a delayed version of the true trajectory.

### PatchTST Multi-Step: RMSE unchanged, event detection improves

PatchTST Multi-Step produces nearly the same RMSE as the baseline (39.051 vs 39.271)
and a similar lag_rmse (14.528 vs 14.213). Δ* is unchanged (-54.6 vs -55.3 min median).
However, event detection improves more clearly: F1=0.0716 vs 0.0609 (+18%) and F2=0.0851
vs 0.0742 (+15%) at tau=15 min.

The event improvement for PatchTST is larger than for LSTM despite the minimal RMSE
change. This suggests that multi-step training helps PatchTST predict threshold crossings
more precisely in timing, consistent with the tau-sweep finding that PatchTST Multi-Step
leads at tau=15 min. Supervising all 12 steps forces the model to learn the shape of the
trajectory closer to the event, which improves detection timing even when RMSE is unchanged.

### Neither model reduces the time-shift substantially

The central hypothesis of Objective 2 is that intermediate supervision reduces accumulated
timing drift. The results do not support this for either architecture: Δ* changes by less
than 2 minutes for LSTM and less than 1 minute for PatchTST. The time-shift arises because
the model learns to produce predictions close to the current glucose level rather than the
true 60-minute future value. Adding more horizon steps to the loss function does not change
this incentive, since the model can still minimise the multi-step MSE by predicting a
smooth, slightly shifted version of the input trajectory.

---

## References

[3] Pattern Recognition Group, University of Bern. Glucose Prediction Proposal.
    Internal unpublished manuscript.
