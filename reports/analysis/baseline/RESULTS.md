# Baseline: Standard MSE Loss

**Branch:** `main`
**Loss function:** Mean Squared Error (MSE)
**Models:** LSTM and PatchTST with Optuna hyperparameter tuning

---

## 1. Approach

The baseline trains patient-specific LSTM and PatchTST models to predict glucose 60 minutes
ahead using standard pointwise MSE as the training objective. Each model receives a lookback
window of 24 steps (120 min) and predicts a single value at h = 11 (the 60-min horizon).
Hyperparameters are tuned via Optuna (50 trials, patient 85202 as representative patient).

This baseline serves as the reference for all extension objectives. Every design choice —
architecture, Optuna search space, training procedure, evaluation protocol — is held fixed
across all three objectives. Only the loss function changes in the extension branches.

---

## 2. Hyperparameters (Optuna, 50 trials, patient 85202)

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

## 3. Summary Results

Event metrics at tau = 3 steps (15 min), threshold = 70 mg/dL. Δ* measured
within ±12 steps (±60 min) by minimising RMSE after shift correction.

| Model | RMSE | lag_rmse | Δ* mean (min) | Δ* median (min) | Δ* std (min) | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| **LSTM MSE** | **36.089** | **21.649** | **-51.2** | **-52.5** | **9.1** | **27.118** | 0.0104 | 0.0036 | 0.0054 |
| **PatchTST MSE** | **39.271** | **14.213** | **-55.3** | **-55.0** | **3.1** | **29.001** | 0.0482 | 0.0888 | 0.0609 |

Δ* is the correction lag in minutes that minimises RMSE(y_true, shift(y_pred, k))
within ±12 steps (±60 min). A negative value means the prediction is late by that
many minutes. For a 60-minute forecast, Δ* ≈ -55 min means the model effectively
predicts only 5 minutes into the future rather than 60 minutes. Both models show
a strong and consistent systematic time-shift, which is the central problem addressed
by all three objectives [3].

---

## 4. Per-Patient Results — LSTM MSE

| Patient | RMSE | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 85101 | 21.001 | 15.866 | 0.000 | 0.000 | 0.000 |
| 85102 | 43.042 | 33.431 | 0.000 | 0.000 | 0.000 |
| 85104 | 28.091 | 20.737 | 0.000 | 0.000 | 0.000 |
| 85105 | 32.243 | 19.729 | 0.375 | 0.130 | 0.194 |
| 85106 | 32.100 | 23.770 | 0.000 | 0.000 | 0.000 |
| 85107 | 27.134 | 20.138 | 0.000 | 0.000 | 0.000 |
| 85111 | 31.013 | 24.121 | 0.000 | 0.000 | 0.000 |
| 85112 | 52.320 | 38.069 | 0.000 | 0.000 | 0.000 |
| 85116 | 47.934 | 33.696 | 0.000 | 0.000 | 0.000 |
| 85117 | 32.822 | 25.607 | 0.000 | 0.000 | 0.000 |
| 85118 | 31.368 | 22.979 | 0.000 | 0.000 | 0.000 |
| 85119 | 37.351 | 27.751 | 0.000 | 0.000 | 0.000 |
| 85120 | 38.674 | 29.770 | 0.000 | 0.000 | 0.000 |
| 85121 | 32.744 | 23.467 | 0.000 | 0.000 | 0.000 |
| 85122 | 46.747 | 35.116 | 0.000 | 0.000 | 0.000 |
| 85123 | 31.252 | 24.406 | 0.000 | 0.000 | 0.000 |
| 85124 | 39.072 | 28.928 | 0.000 | 0.000 | 0.000 |
| 85125 | 35.220 | 25.797 | 0.000 | 0.000 | 0.000 |
| 85126 | 26.468 | 20.202 | 0.000 | 0.000 | 0.000 |
| 85201 | 39.173 | 30.635 | 0.000 | 0.000 | 0.000 |
| 85202 | 34.820 | 24.647 | 0.000 | 0.000 | 0.000 |
| 85204 | 51.689 | 38.399 | 0.000 | 0.000 | 0.000 |
| 85207 | 35.778 | 28.670 | 0.000 | 0.000 | 0.000 |
| 85208 | 27.953 | 20.858 | 0.000 | 0.000 | 0.000 |
| 85209 | 35.882 | 28.287 | 0.000 | 0.000 | 0.000 |
| 85211 | 33.535 | 25.466 | 0.000 | 0.000 | 0.000 |
| 85214 | 39.772 | 31.978 | 0.000 | 0.000 | 0.000 |
| 85215 | 26.573 | 20.257 | 0.000 | 0.000 | 0.000 |
| 85216 | 32.914 | 25.003 | 0.000 | 0.000 | 0.000 |
| 85217 | 45.421 | 35.230 | 0.000 | 0.000 | 0.000 |
| 85218 | 38.572 | 28.802 | 0.000 | 0.000 | 0.000 |
| 85219 | 34.077 | 23.959 | 0.000 | 0.000 | 0.000 |
| 85220 | 37.545 | 29.863 | 0.000 | 0.000 | 0.000 |
| 85221 | 44.211 | 33.107 | 0.000 | 0.000 | 0.000 |
| 85222 | 44.725 | 35.169 | 0.000 | 0.000 | 0.000 |
| 85224 | 29.954 | 22.323 | 0.000 | 0.000 | 0.000 |
| **Mean** | **36.089** | **27.118** | **0.0104** | **0.0036** | **0.0054** |

---

## 5. Per-Patient Results — PatchTST MSE

| Patient | RMSE | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 85101 | 21.318 | 15.783 | 0.000 | 0.000 | 0.000 |
| 85102 | 49.170 | 36.740 | 0.000 | 0.000 | 0.000 |
| 85104 | 32.856 | 23.075 | 0.000 | 0.000 | 0.000 |
| 85105 | 41.071 | 30.386 | 0.034 | 0.087 | 0.049 |
| 85106 | 34.701 | 25.420 | 0.125 | 0.167 | 0.143 |
| 85107 | 29.412 | 21.615 | 0.000 | 0.000 | 0.000 |
| 85111 | 33.493 | 26.386 | 0.000 | 0.000 | 0.000 |
| 85112 | 52.855 | 39.324 | 0.000 | 0.000 | 0.000 |
| 85116 | 51.207 | 37.633 | 0.000 | 0.000 | 0.000 |
| 85117 | 36.576 | 27.873 | 0.000 | 0.000 | 0.000 |
| 85118 | 34.034 | 23.257 | 0.034 | 0.071 | 0.047 |
| 85119 | 41.599 | 29.884 | 0.000 | 0.000 | 0.000 |
| 85120 | 41.522 | 31.696 | 0.000 | 0.000 | 0.000 |
| 85121 | 30.974 | 22.245 | 0.000 | 0.000 | 0.000 |
| 85122 | 52.569 | 39.339 | 0.000 | 0.000 | 0.000 |
| 85123 | 31.660 | 24.127 | 0.143 | 0.286 | 0.190 |
| 85124 | 42.698 | 30.997 | 0.000 | 0.000 | 0.000 |
| 85125 | 39.487 | 27.228 | 0.000 | 0.000 | 0.000 |
| 85126 | 29.261 | 21.160 | 0.000 | 0.000 | 0.000 |
| 85201 | 46.348 | 35.429 | 0.000 | 0.000 | 0.000 |
| 85202 | 36.619 | 24.478 | 0.200 | 0.500 | 0.286 |
| 85204 | 58.915 | 43.002 | 0.045 | 0.167 | 0.071 |
| 85207 | 39.554 | 30.117 | 0.167 | 0.200 | 0.182 |
| 85208 | 30.761 | 23.665 | 0.000 | 0.000 | 0.000 |
| 85209 | 36.998 | 27.809 | 0.054 | 0.133 | 0.077 |
| 85211 | 35.376 | 26.596 | 0.200 | 0.250 | 0.222 |
| 85214 | 41.773 | 30.931 | 0.500 | 1.000 | 0.667 |
| 85215 | 30.448 | 22.881 | 0.000 | 0.000 | 0.000 |
| 85216 | 35.056 | 25.412 | 0.000 | 0.000 | 0.000 |
| 85217 | 49.603 | 36.805 | 0.000 | 0.000 | 0.000 |
| 85218 | 40.351 | 30.255 | 0.067 | 0.100 | 0.080 |
| 85219 | 36.134 | 26.602 | 0.000 | 0.000 | 0.000 |
| 85220 | 38.911 | 29.058 | 0.125 | 0.111 | 0.118 |
| 85221 | 47.362 | 33.948 | 0.000 | 0.000 | 0.000 |
| 85222 | 49.939 | 38.668 | 0.040 | 0.125 | 0.061 |
| 85224 | 33.146 | 24.198 | 0.000 | 0.000 | 0.000 |
| **Mean** | **39.271** | **29.001** | **0.0482** | **0.0888** | **0.0609** |

---

## 6. Analysis and Interpretation

### RMSE: LSTM outperforms PatchTST

The LSTM (RMSE = 36.089) achieves lower pointwise error than PatchTST (RMSE = 39.271) under
standard MSE training. This is consistent with findings in the CGM forecasting literature:
recurrent models trained with MSE naturally optimise for the exact pointwise value at the target
horizon and can outperform attention-based models that attend to longer-range patterns.

### Hypoglycemia detection: near zero for LSTM

The baseline LSTM fails almost entirely at hypoglycemia detection (F1 = 0.0054, recall = 0.36%).
This is a direct consequence of the class imbalance: hypoglycemia events (BG < 70 mg/dL) are
rare in CGM data, and MSE minimisation incentivises the model to predict the mean glucose
trajectory. The model learns to output stable mid-range predictions that minimise aggregate
squared error, systematically ignoring the low-glucose tails.

PatchTST shows substantially higher hypoglycemia recall (8.9%) and F1 (0.061) at baseline.
This likely reflects the attention mechanism's ability to capture longer-range descent patterns
in the lookback window. However, even PatchTST's baseline event detection is clinically
insufficient.

### Time-shift problem

Both models exhibit a characteristic time-shift artefact: predictions reproduce the shape of
the glucose curve but with a delay. This is the central problem addressed by Objectives 1 and 2.
The time-shift means that even when the model predicts a hypoglycemia event, it may predict it
too late for a clinically useful intervention — motivating the use of alignment-aware losses.

### Role as controlled reference

All design decisions (architecture, Optuna search space, patient splits, evaluation at h = 11)
are locked at baseline values and unchanged in extension branches. This ensures that any
difference in results between objectives is attributable solely to the change in loss function.
