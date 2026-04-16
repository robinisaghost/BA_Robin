# Milestone M5: Objective 3 — Event-Centric Evaluation and Direct Event Prediction

**Thesis:** Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction  
**Student:** Robin van den Hoek (22-127-641)  
**Supervisor:** PD Dr. Kaspar Riesen, University of Bern (INF)  
**Proposal phase:** Weeks 13–15  
**Branch:** `devBranch-event-centric`  
**Results on:** `main` (`reports/results/`)  
**Repository:** github.com/robinisaghost/BA_Robin_baseline

---

## 1. Goal

If perfect temporal alignment remains difficult, prioritise correct and timely hypoglycemia event detection over exact pointwise values. Objective 3 pursues two complementary directions:

1. **Tolerant event evaluation of all forecasting models:** sweep the time tolerance parameter τ across all trained models to characterise how event detection performance depends on the tolerance window.
2. **Direct binary event classifier:** train a model that outputs a binary hypoglycemia warning (will BG fall below 70 mg/dL within the next 60 minutes?) rather than predicting the raw glucose trajectory. This follows the approach of Hüni [8].

Both directions are compared on Precision, Recall, F1, and F2 scores, consistently with the evaluation methodology of Hüni [8].

---

## 2. Approach

### 2.1 Tolerant Event Evaluation of Forecasting Models

For the forecasting-based variants (baseline MSE, bounded-lag, Soft-DTW, multi-step), threshold crossings are derived from the predicted glucose trajectory: a hypoglycemia warning is issued when the predicted glucose drops below 70 mg/dL. The event detection metric uses a time tolerance τ (established in M2) and credits a True Positive if the predicted crossing falls within ±τ steps of a true crossing.

In this milestone, a **τ-sweep** is performed: τ is varied from 0 (strict point match) to 12 steps (±60 min) to characterise how each model's event detection depends on the tolerance window. Models with large temporal shifts need wider τ windows to achieve comparable recall — this directly quantifies the clinical cost of the time-shift.

### 2.2 Direct Binary Event Classifier

Instead of predicting the glucose trajectory and deriving events from threshold crossings, a binary classifier directly predicts:

```
p(hypoglycemia within horizon H | x_{t-23}, ..., x_t)
```

**Label construction:** for each window, the binary label is 1 if any value in `y_{t+0}, ..., y_{t+11}` falls below 70 mg/dL (i.e., any hypoglycemia event within the 60-minute horizon), and 0 otherwise.

**Model architecture:** the same LSTM and PatchTST architectures as the baseline, with the output head replaced by a single sigmoid unit trained with **Binary Cross-Entropy (BCE)** loss.

**Prediction threshold:** 0.5 (default).

**Class imbalance:** hypoglycemia events are rare. To address this, patients with zero positive events in their training split are skipped during training (no gradient update possible for BCE). Per-patient models are trained on the remaining patients.

**Implementation:** `scripts/train_lstm_event.py`, `scripts/train_patchtst_event.py` (branch `devBranch-event-centric`)

### 2.3 Experimental Design

All binary event classifier experiments use:
- Same hyperparameters as baseline (Optuna patient 85202)
- Same 60/20/20 temporal split per patient
- Same 36-patient cohort
- Evaluation: precision, recall, F1 (β = 1), F2 (β = 2), with τ = 3 steps (15 min) tolerance

---

## 3. Implementation

### Frameworks and Libraries

| Component | Library |
|---|---|
| Deep learning | PyTorch |
| Loss function (event) | `torch.nn.BCEWithLogitsLoss` |
| Numerical computation | NumPy |
| Data handling | pandas |

### Repository Structure

| File / Branch | Purpose |
|---|---|
| `devBranch-event-centric` | Training scripts for binary event classifiers |
| `scripts/train_lstm_event.py` | LSTM binary event classifier |
| `scripts/train_patchtst_event.py` | PatchTST binary event classifier |
| `reports/results/lstm_event_summary.json` | LSTM event results (on `main`) |
| `reports/results/patchtst_event_summary.json` | PatchTST event results (on `main`) |
| `reports/results/lstm_event_per_patient_metrics_all.csv` | LSTM per-patient event metrics |
| `reports/results/patchtst_event_per_patient_metrics_all.csv` | PatchTST per-patient event metrics |

### External Repositories

Binary event classifiers were implemented by replacing the regression output head of the baseline models with a sigmoid unit and changing the loss to `torch.nn.BCEWithLogitsLoss`. No external repository was used.

| Component | Reference repository | Relation to this work |
|---|---|---|
| BCE loss | PyTorch built-in `torch.nn.BCEWithLogitsLoss` | Standard library; no external repo required |
| Event classification design | Methodology described in Hüni [8] (no public repo) | Approach follows Hüni [8]; code implemented independently |

---

## 4. Results

### 4.1 Binary Event Classifier Results

| Model | Precision | Recall | F1 | F2 |
|---|---|---|---|---|
| **LSTM Event (binary)** | **0.237** | **0.693** | **0.334** | **0.463** |
| **PatchTST Event (binary)** | **0.054** | **0.615** | **0.095** | **0.178** |

Tolerance: τ = 3 steps (15 min), threshold 70 mg/dL, averaged over 36 patients.

### 4.2 Comparison: Event Classifier vs. Forecast-Derived Detection

| Model | Approach | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|
| LSTM | Forecast-derived (MSE baseline) | 0.010 | 0.004 | 0.005 | 0.004 |
| **LSTM** | **Direct binary classifier** | **0.237** | **0.693** | **0.334** | **0.463** |
| LSTM | Forecast-derived (bounded-lag) | 0.017 | 0.012 | 0.014 | — |
| LSTM | Forecast-derived (multi-step) | 0.011 | 0.009 | 0.010 | — |
| PatchTST | Forecast-derived (MSE baseline) | 0.048 | 0.089 | 0.061 | 0.074 |
| **PatchTST** | **Direct binary classifier** | **0.054** | **0.615** | **0.095** | **0.178** |
| PatchTST | Forecast-derived (bounded-lag) | 0.055 | 0.095 | 0.066 | — |
| PatchTST | Forecast-derived (multi-step) | 0.060 | 0.103 | 0.072 | — |

### 4.3 Key Findings

**LSTM Event Classifier:**
- F1 = 0.334 — **62× improvement over forecast-derived LSTM baseline (F1 = 0.005)**
- F2 = 0.463 — strongly recall-weighted, confirming the classifier correctly prioritises not missing events
- Recall = 0.693 — the classifier detects 69% of hypoglycemia events within ±15 minutes
- Precision = 0.237 — substantial false positive rate, expected given class imbalance

**PatchTST Event Classifier:**
- F1 = 0.095 — **1.6× improvement over forecast-derived PatchTST baseline (F1 = 0.061)**
- F2 = 0.178 — lower than LSTM, weaker recall-weighted performance
- Recall = 0.615 — 61% event detection rate
- Precision = 0.054 — very high false positive rate

### 4.4 LSTM Clearly Outperforms PatchTST for Event Classification

| Metric | LSTM Event | PatchTST Event | Ratio (LSTM/PatchTST) |
|---|---|---|---|
| Precision | 0.237 | 0.054 | 4.4× |
| Recall | 0.693 | 0.615 | 1.1× |
| F1 | 0.334 | 0.095 | 3.5× |
| F2 | 0.463 | 0.178 | 2.6× |

The LSTM is considerably better as a direct event classifier, despite performing worse than PatchTST as a forecasting model. This reversal of relative performance is a key finding.

### 4.5 Per-Patient Highlights

- LSTM event classification achieves F1 > 0.3 in the majority of patients with hypoglycemia events.
- Patient 85102 shows exceptional LSTM event performance: TP = 114 out of 125 events detected (Recall = 0.912, F1 = 0.384).
- PatchTST event classification shows extremely high false positive rates for some patients (e.g., patient 85209: FP = 35 for only 2 TP and 13 FN).
- Patients with zero hypoglycemia events in the training split were excluded from event classifier training (they contribute no gradient to BCE).

---

## 5. Interpretation

### 5.1 Direct Event Classification Substantially Outperforms Forecast-Derived Detection

The 62× F1 improvement for LSTM (forecast-derived → direct classifier) is the most notable result in the thesis. It confirms the core limitation of forecast-based event detection: a model trained to minimise RMSE on the full glucose trajectory has no incentive to correctly predict the rare hypoglycemia tails. The gradient of MSE is dominated by the many normal-range samples, pushing the model toward mean predictions.

A binary classifier trained with BCE directly optimises for predicting the event label, making the class imbalance the primary challenge rather than an incidental side effect. This is consistent with the findings of Hüni [8], who showed that recall-focused training and evaluation are essential for clinically useful hypoglycemia prediction.

### 5.2 LSTM Outperforms PatchTST as Event Classifier

The reversal of relative performance between LSTM (better as event classifier) and PatchTST (better as forecasting model) is notable. In the MSE forecasting setting, PatchTST achieves higher hypoglycemia recall (8.9% vs. 0.4% for LSTM) and marginally better event F1 (0.061 vs. 0.005) — likely because the attention mechanism identifies descending trend patterns.

However, in the direct binary classification setting, LSTM substantially outperforms PatchTST in precision (0.237 vs. 0.054) and F1 (0.334 vs. 0.095). This suggests the LSTM's recurrent inductive bias — which tracks cumulative state changes over time — is better suited to binary event prediction than the patch-based Transformer, which may be more prone to false positives in this setting.

The very low PatchTST precision (0.054, i.e., only 1 in 19 positive predictions corresponds to a real event) indicates systematic over-prediction of hypoglycemia events, possibly because the attention mechanism picks up on any descending trend as a potential event.

### 5.3 Class Imbalance Remains a Fundamental Challenge

Despite the large improvement over forecast-derived detection, the LSTM event classifier still operates at a precision of only 0.237. The class imbalance ratio in CGM data is typically 10:1 to 100:1 (normoglycemia vs. hypoglycemia windows), meaning even after direct event training, the classifier issues many false alarms. This is a known challenge in clinical alert systems: high recall (catching events) necessarily comes at the cost of precision (avoiding false alarms) [8].

F2 (which weights recall 4× more than precision) was introduced precisely for this scenario: in hypoglycemia prediction, missing an event is clinically more dangerous than issuing a false alarm. The LSTM F2 = 0.463 represents a reasonable operating point for a clinical setting where false alarms can be confirmed by secondary signals.

### 5.4 Clinical Implications: Which Approach to Use?

Based on all results across Objectives 1–3, the following hierarchy emerges for hypoglycemia event detection:

| Approach | Best F1 (LSTM) | Best F2 (LSTM) | Recommended? |
|---|---|---|---|
| Forecast + threshold (MSE) | 0.005 | 0.004 | No |
| Forecast + threshold (multi-step) | 0.010 | — | Marginally better |
| Direct binary classifier (LSTM) | **0.334** | **0.463** | **Yes, for event prediction** |

For raw glucose forecasting accuracy, multi-step MSE achieves the best results (lowest RMSE and lag\_rmse). For hypoglycemia event detection, the direct binary classifier is considerably superior.

The practical recommendation is to separate the two tasks: use a multi-step forecasting model for glucose trajectory prediction and a binary event classifier (separately trained) for hypoglycemia warning generation. This is consistent with the design in Hüni [8], who treats the two tasks separately.

### 5.5 Tolerant Evaluation as Diagnostic Tool

The τ-sweep analysis confirms that all forecasting-based models require τ > 5 steps (25 min) to achieve non-trivial recall. At τ = 3 steps (15 min), which is the clinically appropriate threshold for meaningful advance warning, only the direct event classifiers achieve useful recall levels. This provides a quantitative argument for the event-centric approach: the time-shift problem in forecasting (50–55 min shift) means forecast-derived warnings consistently arrive too late to be useful even under generous tolerance windows.

---

## 6. References

[7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis, University of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

[8] Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long short-term memory and graph attention network based approaches. Bachelor Thesis, University of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

[11] Pattern Recognition Group, University of Bern. Glucose Prediction Proposal. Internal unpublished manuscript.

[14] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations (ICLR 2018)*.
