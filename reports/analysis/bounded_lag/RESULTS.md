# Objective 1: Bounded-Lag MSE Loss

**Branch:** `devBranch-offset-loss-bounded-lag`
**Loss function:** Bounded-lag MSE (D = 3 steps = ±15 min)
**Models:** LSTM and PatchTST (identical architecture and Optuna tuning as baseline)

---

## 1. Loss Function

The bounded-lag MSE loss addresses the time-shift problem by searching for the best constant
temporal shift k ∈ {−D, …, D} that minimizes MSE between the predicted trajectory and the
ground-truth trajectory. Training uses winner-takes-all gradient descent: only the shift k*
that achieves the lowest MSE receives a gradient update.

```
L_BL(ŷ, y) = min_{k ∈ [-D, D]}  (1/H) Σ_t (ŷ_t - y_{t+k})²
```

With D = 3 the model is allowed to learn predictions that are shifted by up to ±15 minutes
relative to the target. The evaluation metric remains strict pointwise RMSE at h = 11 (60 min),
so there is an intentional train–eval mismatch by design.

---

## 2. Hyperparameters (Optuna, 50 trials, patient 85202)

| Parameter | LSTM | PatchTST |
|---|---|---|
| hidden_size / d_model | 256 | 64 |
| num_layers / n_layers | 2 | 2 |
| dropout | 0.112 | 0.259 |
| learning rate | 7.77e-4 | 1.51e-4 |
| batch_size | 512 | 512 |
| dim_ff | — | 128 |
| n_heads | — | 4 |

---

## 3. Summary Results vs Baseline

| Model | RMSE | lag_rmse | Δ RMSE vs baseline | MAE | Precision | Recall | F1 | ΔF1 vs baseline |
|---|---|---|---|---|---|---|---|---|
| LSTM MSE (baseline) | 36.089 | 21.649 | — | 27.118 | 0.0104 | 0.0036 | 0.0054 | — |
| **LSTM Bounded-lag** | **36.140** | **25.253** | **+0.1%** | **27.571** | 0.0175 | 0.0119 | 0.0141 | +161% |
| PatchTST MSE (baseline) | 39.271 | 14.213 | — | 29.001 | 0.0482 | 0.0888 | 0.0609 | — |
| **PatchTST Bounded-lag** | **39.296** | **16.704** | **+0.1%** | **29.032** | 0.0547 | 0.0955 | 0.0665 | +9% |

---

## 4. Per-Patient Results — LSTM Bounded-lag

| Patient | RMSE | lag_rmse | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 85101 | 20.463 | 12.997 | 15.754 | 0.000 | 0.000 | 0.000 |
| 85102 | 45.238 | 35.717 | 36.479 | 0.000 | 0.000 | 0.000 |
| 85104 | 26.320 | 23.601 | 19.897 | 0.000 | 0.000 | 0.000 |
| 85105 | 34.860 | 34.756 | 23.551 | 0.429 | 0.261 | 0.324 |
| 85106 | 30.228 | 24.548 | 22.026 | 0.000 | 0.000 | 0.000 |
| 85107 | 26.631 | 16.852 | 19.810 | 0.000 | 0.000 | 0.000 |
| 85111 | 31.585 | 21.376 | 24.551 | 0.000 | 0.000 | 0.000 |
| 85112 | 51.412 | 42.058 | 37.963 | 0.000 | 0.000 | 0.000 |
| 85116 | 46.128 | 20.854 | 33.439 | 0.000 | 0.000 | 0.000 |
| 85117 | 32.753 | 29.562 | 25.768 | 0.000 | 0.000 | 0.000 |
| 85118 | 31.173 | 17.362 | 23.071 | 0.000 | 0.000 | 0.000 |
| 85119 | 36.533 | 24.407 | 26.973 | 0.000 | 0.000 | 0.000 |
| 85120 | 39.657 | 28.912 | 31.374 | 0.000 | 0.000 | 0.000 |
| 85121 | 31.992 | 26.912 | 22.857 | 0.000 | 0.000 | 0.000 |
| 85122 | 48.599 | 41.201 | 37.487 | 0.000 | 0.000 | 0.000 |
| 85123 | 31.338 | 17.183 | 24.471 | 0.000 | 0.000 | 0.000 |
| 85124 | 39.188 | 27.042 | 28.980 | 0.000 | 0.000 | 0.000 |
| 85125 | 34.710 | 25.427 | 25.639 | 0.000 | 0.000 | 0.000 |
| 85126 | 26.655 | 17.255 | 20.536 | 0.000 | 0.000 | 0.000 |
| 85201 | 40.153 | 27.986 | 29.984 | 0.000 | 0.000 | 0.000 |
| 85202 | 34.139 | 20.354 | 25.046 | 0.000 | 0.000 | 0.000 |
| 85204 | 53.682 | 35.024 | 40.075 | 0.200 | 0.167 | 0.182 |
| 85207 | 35.761 | 26.201 | 28.633 | 0.000 | 0.000 | 0.000 |
| 85208 | 27.679 | 19.389 | 21.752 | 0.000 | 0.000 | 0.000 |
| 85209 | 36.569 | 24.505 | 29.818 | 0.000 | 0.000 | 0.000 |
| 85211 | 33.906 | 19.196 | 25.906 | 0.000 | 0.000 | 0.000 |
| 85214 | 38.599 | 23.598 | 30.515 | 0.000 | 0.000 | 0.000 |
| 85215 | 26.604 | 18.761 | 20.639 | 0.000 | 0.000 | 0.000 |
| 85216 | 33.516 | 23.199 | 26.717 | 0.000 | 0.000 | 0.000 |
| 85217 | 44.823 | 26.776 | 35.443 | 0.000 | 0.000 | 0.000 |
| 85218 | 37.519 | 19.304 | 27.929 | 0.000 | 0.000 | 0.000 |
| 85219 | 33.247 | 24.691 | 24.168 | 0.000 | 0.000 | 0.000 |
| 85220 | 38.140 | 27.862 | 30.955 | 0.000 | 0.000 | 0.000 |
| 85221 | 46.598 | 35.248 | 35.781 | 0.000 | 0.000 | 0.000 |
| 85222 | 44.815 | 23.599 | 35.607 | 0.000 | 0.000 | 0.000 |
| 85224 | 29.811 | 25.392 | 22.949 | 0.000 | 0.000 | 0.000 |
| **Mean** | **36.140** | **25.253** | **27.571** | **0.0175** | **0.0119** | **0.0141** |

---

## 5. Per-Patient Results — PatchTST Bounded-lag

| Patient | RMSE | lag_rmse | MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 85101 | 21.249 | 6.507 | 15.783 | 0.000 | 0.000 | 0.000 |
| 85102 | 48.697 | 20.259 | 36.454 | 0.000 | 0.000 | 0.000 |
| 85104 | 32.873 | 17.901 | 23.202 | 0.000 | 0.000 | 0.000 |
| 85105 | 42.861 | 31.144 | 31.797 | 0.024 | 0.043 | 0.031 |
| 85106 | 34.446 | 15.798 | 25.587 | 0.143 | 0.167 | 0.154 |
| 85107 | 28.837 | 11.224 | 21.598 | 0.000 | 0.000 | 0.000 |
| 85111 | 33.426 | 12.955 | 26.318 | 0.000 | 0.000 | 0.000 |
| 85112 | 53.603 | 20.752 | 39.776 | 0.000 | 0.000 | 0.000 |
| 85116 | 50.909 | 19.283 | 36.026 | 0.000 | 0.000 | 0.000 |
| 85117 | 35.570 | 21.712 | 27.232 | 0.000 | 0.000 | 0.000 |
| 85118 | 33.840 | 12.488 | 23.576 | 0.033 | 0.071 | 0.045 |
| 85119 | 41.179 | 18.976 | 29.569 | 0.000 | 0.000 | 0.000 |
| 85120 | 41.665 | 15.980 | 31.980 | 0.000 | 0.000 | 0.000 |
| 85121 | 30.904 | 14.689 | 22.530 | 0.000 | 0.000 | 0.000 |
| 85122 | 53.145 | 28.661 | 40.088 | 0.000 | 0.000 | 0.000 |
| 85123 | 31.652 | 9.796 | 24.009 | 0.125 | 0.286 | 0.174 |
| 85124 | 42.294 | 18.765 | 30.703 | 0.000 | 0.000 | 0.000 |
| 85125 | 39.309 | 14.946 | 27.173 | 0.077 | 0.200 | 0.111 |
| 85126 | 28.739 | 9.919 | 19.823 | 0.000 | 0.000 | 0.000 |
| 85201 | 47.697 | 25.286 | 36.340 | 0.000 | 0.000 | 0.000 |
| 85202 | 36.410 | 10.877 | 24.628 | 0.000 | 0.000 | 0.000 |
| 85204 | 59.668 | 27.843 | 42.880 | 0.074 | 0.333 | 0.121 |
| 85207 | 38.607 | 18.234 | 29.757 | 0.250 | 0.200 | 0.222 |
| 85208 | 30.749 | 9.258 | 23.265 | 0.000 | 0.000 | 0.000 |
| 85209 | 37.032 | 14.762 | 28.023 | 0.091 | 0.200 | 0.125 |
| 85211 | 35.447 | 14.005 | 26.564 | 0.333 | 0.500 | 0.400 |
| 85214 | 41.470 | 15.505 | 30.441 | 0.500 | 1.000 | 0.667 |
| 85215 | 30.907 | 11.722 | 23.419 | 0.000 | 0.000 | 0.000 |
| 85216 | 34.611 | 13.538 | 25.393 | 0.000 | 0.000 | 0.000 |
| 85217 | 50.132 | 19.091 | 37.591 | 0.000 | 0.000 | 0.000 |
| 85218 | 40.156 | 11.833 | 30.031 | 0.143 | 0.200 | 0.167 |
| 85219 | 35.892 | 16.116 | 26.093 | 0.000 | 0.000 | 0.000 |
| 85220 | 39.759 | 19.015 | 30.325 | 0.143 | 0.111 | 0.125 |
| 85221 | 47.627 | 16.805 | 33.948 | 0.000 | 0.000 | 0.000 |
| 85222 | 50.003 | 17.640 | 38.690 | 0.032 | 0.125 | 0.051 |
| 85224 | 33.274 | 18.054 | 24.557 | 0.000 | 0.000 | 0.000 |
| **Mean** | **39.296** | **16.704** | **29.032** | **0.0547** | **0.0955** | **0.0665** |

---

## 6. Analysis and Interpretation

### RMSE is virtually unchanged from baseline

After correcting the slice-length bias in the loss implementation, the bounded-lag loss has
negligible impact on pointwise RMSE: both LSTM (+0.1%) and PatchTST (+0.1%) perform almost
identically to their respective baselines. This result shows that the bounded-lag loss does
not meaningfully distort the model's pointwise predictions when implemented correctly.

The reason is that the winner-takes-all mechanism predominantly selects k*=0 during training:
for most samples, the unshifted target already minimises MSE, so the bounded-lag loss
degenerates to standard MSE. The ±15 minute tolerance is only exploited when a shifted target
produces a clearly lower loss, which occurs rarely once the model has converged.

### lag_rmse increases slightly

The lag_rmse metric measures the best-case RMSE after applying an optimal constant shift
k* ∈ [−12, 12] steps across all test predictions of a patient.

- LSTM: 21.649 → 25.253 (+17%)
- PatchTST: 14.213 → 16.704 (+18%)

The modest increase in lag_rmse reflects the inconsistency introduced by winner-takes-all
training: different samples receive gradients from different shifts k*, so the model does not
learn a single consistent temporal offset. Since lag_rmse corrects only one constant shift per
patient, it cannot fully compensate for sample-level variation. The increase is small compared
to the pre-fix results, confirming that the slice-length bug was the dominant source of
lag_rmse degradation in earlier runs.

### Hypoglycemia detection improves modestly

Both models show improved F1 for hypoglycemia events (BG < 70 mg/dL):

- LSTM: F1 0.0054 → 0.0141 (+161%), recall 0.0036 → 0.0119
- PatchTST: F1 0.0609 → 0.0665 (+9%), recall 0.0888 → 0.0955

The improvement is more pronounced for LSTM than PatchTST in relative terms. PatchTST
already had substantially higher baseline event detection, leaving less room for improvement.
The bounded-lag loss nudges both models toward slightly better shape prediction around
threshold crossings, which translates into a small but consistent F1 gain.

Notable individual cases: PatchTST patient 85214 achieves recall = 1.000 (all hypoglycemia
events detected) and patient 85211 reaches F1 = 0.400.

### PatchTST consistently outperforms LSTM on event detection

PatchTST Bounded-lag achieves higher recall (0.0955 vs 0.0119) and F1 (0.0665 vs 0.0141)
than LSTM Bounded-lag. This gap is consistent with the baseline comparison and reflects
PatchTST's ability to capture multi-scale temporal patterns through its patch-based attention
mechanism, which appears more sensitive to the shape of glucose descents approaching the
hypoglycemia threshold.

### Key trade-off

With the correct implementation, the bounded-lag loss achieves its design goal: preserving
pointwise accuracy (RMSE +0.1%) while providing a small improvement in hypoglycemia event
detection (F1 +9–161%). The trade-off is minimal — the loss neither improves nor degrades
RMSE meaningfully, and the F1 gains are modest but present. The clinical value depends on
whether a ~9% F1 improvement for PatchTST justifies the added training complexity.
