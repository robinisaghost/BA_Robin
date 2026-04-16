# Milestone M2: Protocol Fixed and Pipeline Validated

**Thesis:** Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction  
**Student:** Robin van den Hoek (22-127-641)  
**Supervisor:** PD Dr. Kaspar Riesen, University of Bern (INF)  
**Proposal phase:** Weeks 4–6  
**Branch:** `main`  
**Repository:** github.com/robinisaghost/BA_Robin_baseline

---

## 1. Goal

Design, implement, and validate the full evaluation protocol used across all thesis objectives. This milestone covers two distinct components:

1. **Shift measurement pipeline:** a metric quantifying the temporal misalignment (time-shift) between predicted and true glucose trajectories, independent of pointwise error.
2. **Tolerant event evaluation:** a hypoglycemia detection metric that credits predictions within a time tolerance window τ of the true event, following the methodology of Hüni [8].

Both components are validated on the baseline (M1) results and tested against synthetic shifts to confirm correctness. The finalised protocol is then applied identically across all branches (Objectives 1–3), making M2 the methodological foundation of the thesis.

---

## 2. Approach

### 2.1 Motivating the Protocol

Standard pointwise metrics (RMSE, MAE) conflate two distinct sources of error: (a) amplitude error — the model predicts the wrong glucose level — and (b) temporal error — the model predicts the correct trajectory shape but shifted in time. When RMSE is large primarily due to a temporal offset, it gives a misleading picture of model quality: the prediction could still be clinically useful if it arrived at the right time [11].

The thesis addresses this by introducing two supplementary metrics:

- **Best-lag Δ\* (cross-correlation):** quantifies the time-shift directly.
- **Lag-adjusted RMSE (lag\_rmse):** measures the minimum achievable RMSE after compensating for the best constant temporal shift; isolates amplitude error from temporal error.

### 2.2 Best-Lag via Cross-Correlation

The best-lag Δ\* is computed by maximising the normalised cross-correlation between the predicted and true glucose series within a bounded window [−D, D]:

```
Δ* = argmax_{Δ ∈ [-D,D]}  (y_true_shifted · y_pred) / (||y_true|| · ||y_pred||)
```

Both series are mean-centred before the computation. The window D is set to **12 steps (60 minutes)**, matching the prediction horizon. A positive Δ\* indicates the prediction is shifted forward (delayed) relative to ground truth.

**Implementation:** `src/ba_baseline/metrics/metrics.py`, function `best_lag_crosscorr`.

### 2.3 Lag-Adjusted RMSE

The lag-adjusted RMSE (lag\_rmse) finds the shift k ∈ [−D, D] that minimises RMSE after shifting the prediction by k steps:

```
lag_rmse = min_{k ∈ [-D,D]}  RMSE(y_true, shift(y_pred, k))
```

`shift(y_pred, k)` shifts the prediction by k positions, filling vacated positions with NaN (excluded from RMSE computation). The difference

```
shift_penalty = RMSE - lag_rmse
```

isolates the fraction of total error that is purely due to a constant temporal offset. A large shift\_penalty confirms the time-shift hypothesis; a small one indicates the error is dominated by amplitude mismatch.

**Implementation:** `src/ba_baseline/metrics/metrics.py`, functions `best_lag_rmse` and `lag_adjusted_rmse`.

### 2.4 Tolerant Hypoglycemia Event Evaluation

A hypoglycemia event is defined as a threshold crossing from above to below **70 mg/dL** in the glucose series [8]. Standard binary detection at a fixed time point fails when the prediction is temporally shifted: even a correct prediction may be counted as a False Positive if it arrives τ steps too early or late.

The tolerant evaluation assigns True Positive credit to a predicted crossing pc if it falls within [tc − τ, tc + τ] of a true crossing tc. Each true event can be matched at most once.

**Tolerance:** τ = **3 steps (15 minutes)**, consistent with the clinical threshold established by Hüni [8] and the thesis proposal [11].

**Metrics reported:** Precision, Recall, F1 (β = 1), and F2 (β = 2, which weights recall twice as heavily as precision — clinically appropriate because missing a hypoglycemia warning is more dangerous than a false alarm [8]).

**Implementation:** `src/ba_baseline/metrics/metrics.py`, functions `crossing_times` and `event_metrics`.

### 2.5 Synthetic Shift Validation

To validate that the shift measurement is correct, the `shift_1d` utility was applied to the ground-truth series with known offsets (k = 2, 5, 10 steps) and the recovered best-lag was verified to match the input shift. This confirms the measurement infrastructure is free from off-by-one errors or boundary artefacts.

### 2.6 Comparison Against Peer Implementation

As part of M2, a systematic comparison was conducted between this implementation and the LSTM/PatchTST baseline from a parallel master's thesis (ahonongobi/Master-Thesis on GitHub). The comparison revealed three methodological errors in the peer implementation that render its metrics not directly comparable:

1. **RMSE scope:** the peer model reports RMSE averaged over all 12 output steps (5–60 min), whereas this implementation reports the 60-minute RMSE exclusively. Averaging over short horizons masks the time-shift problem, since early steps (5–30 min) have negligible shift.
2. **Data Leakage:** the peer implementation fits a global MinMaxScaler on the entire dataset (train + val + test) before splitting — leaking test statistics into normalisation.
3. **Patient boundary violation:** sliding windows are created over the concatenated time series of all patients, generating ~2500 spurious training samples that combine physiologically unrelated glucose readings from different patients.

These findings validate the methodological choices of this implementation and confirm that the observed time-shift (50–55 minutes) is a genuine artefact, not a measurement error.

Full analysis in `reports/analysis/comparison_baseline_robin_vs_fellow.md`.

---

## 3. Implementation

### Key Source Files

| File | Purpose |
|---|---|
| `src/ba_baseline/metrics/metrics.py` | RMSE, MAE, `best_lag_crosscorr`, `lag_adjusted_rmse`, `crossing_times`, `event_metrics` |
| `src/ba_baseline/data/split.py` | Temporal 60/20/20 split, per-patient, no leakage |
| `src/ba_baseline/data/multi_patient_dataset.py` | Sliding windows within patient boundaries, Z-score normalisation |
| `scripts/compute_full_metrics.py` | Loads `.npz` traces, computes all metrics, writes CSV summaries |
| `scripts/plot_shift_60min.py` | Visualises prediction vs. ground truth with shift annotation |

### External Repositories

All metric and evaluation code was implemented from scratch. No external repository code was used directly.

| Component | Reference repository | Relation to this work |
|---|---|---|
| Cross-correlation best-lag | Standard NumPy (`numpy.correlate`) | Library function; no external repo required |
| Tolerant event evaluation | Methodology from Hüni [8] | Re-implemented from scratch based on the thesis description; peer repo `github.com/ahonongobi/Master-Thesis` consulted for comparison only |

### Frameworks

| Component | Library |
|---|---|
| Numerical computation | NumPy |
| Data handling | pandas |
| Visualisation | Matplotlib |

### Parameters Fixed by M2

| Parameter | Value | Rationale |
|---|---|---|
| Lag window D | ±12 steps (±60 min) | Matches prediction horizon |
| Event threshold | 70 mg/dL | Clinical hypoglycemia threshold [8] |
| Time tolerance τ | 3 steps (15 min) | Clinical grace window [8] |
| F2 beta | 2.0 | Recall-weighted, per Hüni [8] |

---

## 4. Results

### 4.1 Shift Measurement on Baseline

| Model | Mean Δ\* (steps) | Mean Δ\* (min) | Std Δ\* (steps) |
|---|---|---|---|
| LSTM | 10.22 | 51.1 | 1.84 |
| PatchTST | 11.00 | 55.0 | 0.63 |

Both models exhibit a near-maximal shift: the prediction lags the ground truth by approximately **10–11 steps (50–55 minutes)** on average, approaching the full 60-minute horizon. The shift is notably consistent across patients (PatchTST std = 0.63 steps = 3 minutes), confirming it is a systemic model property and not noise.

### 4.2 Tolerant Event Detection on Baseline

The event detection metrics at τ = 3 steps (15 min) for the baseline models are reported in M1 and reproduced here for completeness:

| Model | Precision | Recall | F1 | F2 |
|---|---|---|---|---|
| LSTM | 0.010 | 0.004 | 0.005 | 0.004 |
| PatchTST | 0.048 | 0.089 | 0.061 | 0.074 |

The near-zero F1 for LSTM and low F1 for PatchTST under standard MSE training establish the baseline event detection performance against which Objectives 1–3 are compared.

### 4.3 Synthetic Validation

The shift measurement correctly recovered all synthetic shifts (k = 2, 5, 10 steps) applied to the ground-truth series, confirming implementation correctness.

---

## 5. Interpretation

### 5.1 The Time-Shift is a Dominant Error Source

The 51–55 minute shift quantified here explains why RMSE is high even though prediction shapes appear clinically plausible. At h = 11 (60 min), the optimal alignment shift is nearly one full horizon — meaning the model is essentially predicting the current state rather than the future state. This confirms the observation of the Pattern Recognition Group [11] that "prediction shifts can dominate RMSE even when the predicted curve is clinically plausible."

The lag\_rmse metric — introduced as part of this pipeline — makes this decomposition explicit: it isolates the best-case error after correcting for the temporal offset. A large gap between RMSE and lag\_rmse indicates the shift is a major error driver; a small gap means the remaining error is amplitude-based. This decomposition is used in Objectives 1–3 to assess whether alignment-aware training actually reduces the shift, rather than merely redistributing error.

### 5.2 Clinical Significance of the τ-Tolerant Metric

The standard F1 metric without time tolerance severely penalises predictions that are clinically useful but arrive slightly early or late. A hypoglycemia warning issued 10 minutes before the true event onset can still allow a patient to take corrective action (glucose intake, insulin adjustment). The τ = 15-minute tolerance window is consistent with clinical practice guidelines and the evaluation methodology of Hüni [8], providing a fairer measure of clinical utility.

### 5.3 Why F2 is Used

The F2 score weights recall β² = 4 times more than precision. In the context of hypoglycemia detection, a False Negative (missed warning) is clinically more dangerous than a False Positive (unnecessary alert). This prioritisation is consistent with the recall-focused evaluation design of Hüni [8] and is standard in clinical alert systems where the cost of missing an event is asymmetrically higher.

### 5.4 Protocol Stability

The fixed protocol — identical across all branches — ensures that any difference in results between the baseline and Objectives 1–3 is attributable solely to the training change. This ablation-style design follows the guidance of the Pattern Recognition Group [11] and is explicitly stated in the thesis proposal as the core experimental methodology.

---

## 6. References

[7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis, University of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

[8] Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long short-term memory and graph attention network based approaches. Bachelor Thesis, University of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

[11] Pattern Recognition Group, University of Bern. Glucose Prediction Proposal. Internal unpublished manuscript.
