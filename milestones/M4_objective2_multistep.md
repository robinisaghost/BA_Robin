# Milestone M4: Objective 2 — Multi-Step Forecasting

**Thesis:** Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction  
**Student:** Robin van den Hoek (22-127-641)  
**Supervisor:** PD Dr. Kaspar Riesen, University of Bern (INF)  
**Proposal phase:** Weeks 10–12  
**Branch:** `devBranch-multi-step`  
**Results on:** `main` (`reports/results/`)  
**Repository:** github.com/robinisaghost/BA_Robin_baseline

---

## 1. Goal

Train models to predict the full 12-step future trajectory (H = 12 steps = 60 minutes) simultaneously, rather than a single horizon point. The hypothesis is that predicting intermediate steps forces the model to remain "in phase" over the full horizon, reducing the temporal drift that causes the time-shift artefact [11].

Evaluated against the same metrics as all other objectives: RMSE at h = 11 (the 60-minute step), lag\_rmse (best-case RMSE after shift correction), MAE, and hypoglycemia F1/F2. All hyperparameters and the evaluation protocol are identical to the baseline (M1).

---

## 2. Approach

### 2.1 Motivation

In the baseline (M1), the model outputs a single value `ŷ_{t+11}` and is trained exclusively on MSE at that horizon. This is a **direct 60-minute forecasting** setup. The training signal comes only from predicting the 60-minute future; intermediate steps provide no supervision. The model is therefore free to drift in the temporal dimension over the 60-minute gap, which is the root cause of the observed 50–55 minute shift [11].

Multi-step supervision provides **intermediate supervision**: the model is trained to simultaneously predict `ŷ_{t+0}, ŷ_{t+1}, ..., ŷ_{t+11}` (all 12 steps = 60 minutes at 5-minute intervals). The training loss is the mean MSE over all output steps:

```
L_multistep(ŷ, y) = (1/H) Σ_{k=0}^{H-1}  MSE(ŷ_{t+k}, y_{t+k})
```

This forces the model to remain consistent across the full trajectory. A prediction that is correct at h = 5 min (step 0) but shifted at h = 60 min (step 11) would incur high loss at intermediate steps, penalising temporal drift incrementally. The evaluation target remains h = 11 (60 minutes), as in all other objectives.

### 2.2 Multi-Step Forecasting Strategies

Two established multi-step strategies exist in the time-series literature [13]:

- **Recursive (MIMO):** predict one step at a time, feeding each prediction as input to the next. Accumulates errors across steps.
- **Direct multi-output (DIRMO / sequence-to-sequence):** predict all H steps simultaneously from the same input window. No error accumulation between steps.

This thesis uses the **direct multi-output** approach: the model receives the 24-step lookback and outputs all 12 future steps simultaneously. This is the most common approach in deep learning-based time-series forecasting [5] and avoids the compounding error problem of recursive prediction.

### 2.3 Architecture Changes

The only change relative to the baseline is the output dimension: the model now outputs `(batch, H)` instead of `(batch, 1)`. The training loss is averaged over all H = 12 output positions.

**LSTM:** a linear head maps the final hidden state to H = 12 output values in parallel.  
**PatchTST:** the output head maps the patch embedding to H = 12 values; RevIN denormalisation is applied per-step.

All other components (depth, width, dropout, learning rate, batch size) use the baseline hyperparameters from Optuna (patient 85202). This is the same ablation constraint as all other objectives.

### 2.4 Evaluation

Evaluation is performed at **h = 11 only** (the 60-minute step), consistent with all other branches. This means multi-step and single-step models are evaluated on the same target, enabling direct comparison. The intermediate steps are used for training only.

---

## 3. Implementation

### Frameworks and Libraries

| Component | Library |
|---|---|
| Deep learning | PyTorch |
| Numerical computation | NumPy |
| Data handling | pandas |

### Repository Structure

| File / Branch | Purpose |
|---|---|
| `devBranch-multi-step` | Full training scripts and model configuration |
| `scripts/train_lstm_multistep.py` | LSTM multi-step training (on `devBranch-multi-step`) |
| `scripts/train_patchtst_multistep.py` | PatchTST multi-step training (on `devBranch-multi-step`) |
| `reports/results/lstm_multistep_summary.json` | LSTM results (on `main`) |
| `reports/results/patchtst_multistep_summary.json` | PatchTST results (on `main`) |
| `reports/results/lstm_multistep_per_patient_metrics_all.csv` | LSTM per-patient results (on `main`) |
| `reports/results/patchtst_multistep_per_patient_metrics_all.csv` | PatchTST per-patient results (on `main`) |

---

## 4. Results

### 4.1 Summary — All Objectives Compared

| Model | Loss | RMSE (mg/dL) | lag\_rmse (mg/dL) | Shift penalty | MAE (mg/dL) | Hypo F1 |
|---|---|---|---|---|---|---|
| LSTM | MSE (baseline) | 36.089 | — | — | 27.118 | 0.0054 |
| LSTM | Bounded-Lag | 36.140 | 25.253 | 10.887 | 27.571 | 0.0141 |
| LSTM | Soft-DTW | 37.232 | 28.668 | 8.564 | 28.760 | 0.0137 |
| **LSTM** | **Multi-step** | **35.586** | **20.744** | **14.842** | **26.803** | **0.0098** |
| PatchTST | MSE (baseline) | 39.271 | — | — | 29.001 | 0.0609 |
| PatchTST | Bounded-Lag | 39.296 | 16.704 | 22.592 | 29.032 | 0.0665 |
| PatchTST | Soft-DTW | 43.235 | 24.006 | 19.229 | 32.251 | 0.0624 |
| **PatchTST** | **Multi-step** | **39.051** | **14.528** | **24.523** | **28.837** | **0.0716** |

### 4.2 Key Findings

**LSTM Multi-step:**
- RMSE = 35.586 — **lowest RMSE of all LSTM variants**, marginally below baseline (−0.503 mg/dL)
- lag\_rmse = 20.744 — **lowest lag\_rmse of all LSTM variants**
- Shift penalty = RMSE − lag\_rmse = **14.842 mg/dL** — the largest shift penalty observed for LSTM
- MAE = 26.803 — **lowest MAE of all LSTM variants**

**PatchTST Multi-step:**
- RMSE = 39.051 — marginally below baseline (−0.220 mg/dL)
- lag\_rmse = **14.528 — lowest lag\_rmse of all PatchTST variants and the lowest across all models**
- Shift penalty = **24.523 mg/dL** — the largest shift penalty for PatchTST
- MAE = 28.837 — marginally below baseline (−0.164 mg/dL)
- Hypo F1 = **0.0716 — highest of all PatchTST variants**

### 4.3 lag\_rmse Reduction vs. Offset-Aware Losses

| Model | Bounded-Lag lag\_rmse | DTW lag\_rmse | Multi-step lag\_rmse |
|---|---|---|---|
| LSTM | 25.253 | 28.668 | **20.744** |
| PatchTST | 16.704 | 24.006 | **14.528** |

Multi-step achieves the lowest lag\_rmse for both architectures, outperforming both offset-aware loss variants.

### 4.4 Shift Penalty Interpretation

The shift penalty (RMSE − lag\_rmse) measures how much of the pointwise error is attributable to a constant temporal offset. Multi-step achieves both the lowest lag\_rmse and a large shift penalty, meaning:

- The model's best-case error (after shift correction) is the lowest.
- However, a large fraction of the total RMSE remains attributable to temporal offset — the multi-step model has not eliminated the shift; it has become better at the underlying shape while still exhibiting temporal drift.

This is an important nuance: **lower lag\_rmse does not mean the shift is gone** — it means the model's trajectory shape is more accurate after alignment.

### 4.5 Per-Patient Highlights

LSTM multi-step improvements are consistent across patients:
- RMSE decreases or is stable for the majority of patients relative to baseline.
- lag\_rmse is substantially lower for patients with high glucose variability (e.g., patient 85102: lag\_rmse = 29.30 vs. bounded-lag 35.72).
- Hypoglycemia F1 remains near zero for most patients due to class imbalance (unchanged from baseline).

---

## 5. Interpretation

### 5.1 Multi-Step Supervision as Temporal Regularisation

The most notable result of Objective 2 is that multi-step MSE supervision achieves the **lowest lag\_rmse of all methods tested**, despite using only a standard MSE loss. This supports the hypothesis of the Pattern Recognition Group [11]: predicting intermediate steps provides implicit temporal regularisation that keeps the model "in phase" over the full horizon.

Intuitively, the multi-step loss creates a curriculum of increasingly difficult prediction tasks (5 min, 10 min, ..., 60 min). For the loss at intermediate steps (e.g., h = 5 at 25 minutes) to be small, the model's trajectory must be plausible not only at 60 minutes but also at all preceding time points. Any temporal drift that causes a wrong shape at 25 minutes is penalised directly — whereas the single-output baseline receives no such signal.

This is consistent with the theoretical analysis of multi-step forecasting by Taieb & Atiya [13], who show that direct multi-output strategies (predicting all steps simultaneously) can reduce bias in long-horizon predictions by propagating error signal across the full trajectory.

### 5.2 RMSE vs. lag\_rmse Trade-off

Multi-step achieves a better lag\_rmse but a larger shift penalty than bounded-lag for LSTM (14.842 vs. 10.887). This means the multi-step model's best-case error is lower, but a larger fraction of its RMSE is attributable to a remaining temporal offset. The offset-aware bounded-lag loss, by contrast, has a smaller shift penalty — the model is more temporally consistent, but its underlying amplitude error (lag\_rmse) is higher.

Neither approach fully solves the time-shift problem. The shift penalty remaining for multi-step (14.842 mg/dL for LSTM, 24.523 mg/dL for PatchTST) confirms that even with full trajectory supervision, the model still exhibits temporal drift. This is likely because the temporal offset emerges from the physics of glucose dynamics at 60-minute horizons: glucose trends are slow and smooth, and the model's tendency to extrapolate the current trend persists regardless of intermediate supervision.

### 5.3 PatchTST Benefits More Than LSTM

PatchTST's lag\_rmse improvement under multi-step (14.528, a 63% reduction from baseline RMSE of 39.271) is substantially larger than LSTM's (20.744, a 42% reduction from 36.089). This suggests the Transformer architecture benefits more from full-trajectory supervision, possibly because the attention mechanism is better positioned to identify and track multi-scale temporal patterns across all 12 output steps simultaneously.

PatchTST also achieves the best hypoglycemia F1 of all variants (0.0716), suggesting that multi-step supervision marginally improves event detection even under the class imbalance constraint.

### 5.4 Clinical Relevance

From a clinical perspective, the lower lag\_rmse means that after accounting for the residual temporal offset, the multi-step model's glucose predictions are more accurate in amplitude. This is relevant for hypoglycemia warning systems: if the clinician or automated system applies a known constant shift correction (e.g., issue the warning τ\* minutes earlier than the model predicts), the multi-step model would provide the most accurate corrected warning.

However, the near-zero hypoglycemia F1 (0.0098 for LSTM, 0.0716 for PatchTST) shows that multi-step MSE training alone is insufficient for reliable hypoglycemia event detection. Objective 3 addresses this limitation directly.

### 5.5 Conclusion

Multi-step forecasting is the strongest single intervention tested in this thesis for reducing the time-shift artefact, as measured by lag\_rmse. It achieves this without any additional loss engineering, using only standard MSE over a full output trajectory. The result directly validates the hypothesis stated in the proposal [11] and is consistent with the multi-step forecasting literature [13].

---

## 6. References

[5] Lim, B., & Zohren, S. (2020). Time series forecasting with deep learning: A survey. *Philosophical Transactions of the Royal Society A*, 379(2194). https://doi.org/10.1098/rsta.2020.0209

[7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis, University of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

[8] Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long short-term memory and graph attention network based approaches. Bachelor Thesis, University of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

[11] Pattern Recognition Group, University of Bern. Glucose Prediction Proposal. Internal unpublished manuscript.

[13] Taieb, S. B., & Atiya, A. F. (2016). A bias and variance analysis for multistep-ahead time series forecasting. *IEEE Transactions on Neural Networks and Learning Systems*, 27(1), 62–76. https://doi.org/10.1109/TNNLS.2015.2427491
