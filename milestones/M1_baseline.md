# Milestone M1: Baseline Reproduced

**Thesis:** Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction  
**Student:** Robin van den Hoek (22-127-641)  
**Supervisor:** PD Dr. Kaspar Riesen, University of Bern (INF)  
**Proposal phase:** Weeks 1–3  
**Branch:** `main`  
**Repository:** github.com/robinisaghost/BA_Robin_baseline

---

## 1. Goal

Reproduce a clean, methodologically correct baseline for 60-minute blood glucose forecasting using LSTM and PatchTST architectures on the CGM dataset from prior University of Bern work [8]. This baseline serves as the controlled reference point against which all three thesis objectives are compared. Every design decision — architecture, hyperparameter search space, evaluation protocol, patient splits — is locked at baseline values and left unchanged in the extension branches. Only the loss function or training objective changes between branches.

---

## 2. Approach

### 2.1 Dataset

The primary dataset is the CGM dataset used in Hüni [8], consisting of 37 patients with 5-minute continuous glucose monitoring (CGM) recordings [9]. One patient was excluded due to insufficient data, leaving **36 patients** for all experiments. Blood glucose values are measured in mg/dL with a 5-minute sampling interval.

The dataset was obtained from a published clinical trial on automated insulin delivery in adults with type 1 diabetes [9]. Only the CGM signal is used as input; no insulin or meal information is included, consistent with the CGM-only scope of Hüni [8].

### 2.2 Task Formulation

Each model receives a **lookback window of 24 steps (120 minutes)** and predicts a single glucose value at horizon **h = 11 (60 minutes ahead)**, indexed as step index 11 within a 12-step output horizon. This direct 60-minute forecasting setup was identified in internal experiments [11] as the horizon at which time-shift artefacts are clinically most relevant.

Formally, the model maps:

```
x_{t-23}, x_{t-22}, ..., x_t  →  ŷ_{t+11}
```

where each step corresponds to 5 minutes, making x a 2-hour history window and ŷ_{t+11} the glucose prediction 60 minutes into the future.

### 2.3 Data Pipeline

**Splitting:** Each patient's time series is split temporally into train/validation/test sets with a fixed 60/20/20 ratio, consistent with the protocol of Hüni [8]. No shuffling is applied — temporal order is preserved to avoid future data leakage. Patient boundaries are strictly respected: no sliding window crosses from one patient's series into another's.

**Normalisation:** Per-patient Z-score normalisation is applied. The mean and standard deviation are computed exclusively from the training split of each patient and then applied to the validation and test splits [6]. This prevents any form of data leakage through the normalisation statistics.

**Dataset construction:** Sliding windows are created within each patient's split independently. The `MultiPatientWindowDataset` class iterates per patient; each (x, y) sample belongs to exactly one patient's time series.

### 2.4 Models

**LSTM** — A recurrent neural network with Long Short-Term Memory cells [1]. The LSTM receives the 24-step lookback as a sequence, processes it through stacked LSTM layers, and outputs the prediction via a linear head. LSTM is the standard recurrent baseline for CGM forecasting and was used by Hüni [8] as the primary model.

**PatchTST** — A Transformer-based time-series model proposed by Nie et al. [2]. It divides the input sequence into non-overlapping patches, processes them via a Transformer encoder [3], and applies RevIN (Reversible Instance Normalisation [4]) per input window to reduce distribution shift. PatchTST is used as the modern attention-based baseline.

Both models output a single scalar (the 60-minute-ahead prediction) trained with standard **Mean Squared Error (MSE)** loss.

### 2.5 Hyperparameter Tuning

Hyperparameters were tuned via **Optuna** [10] with 50 trials using the **TPE (Tree-structured Parzen Estimator)** sampler. Tuning was performed on a single representative patient (patient 85202) and the resulting hyperparameters were applied to all 36 patients. This approach follows the ablation-style design of the thesis: hyperparameters are fixed once at baseline and held constant across all objective branches, ensuring that differences in results are attributable solely to the change in loss function or training objective.

| Parameter | LSTM | PatchTST |
|---|---|---|
| hidden\_size / d\_model | 256 | 64 |
| num\_layers / n\_layers | 2 | 3 |
| dropout | 0.1585 | 0.0254 |
| learning rate | 9.61 × 10⁻⁴ | 1.21 × 10⁻⁴ |
| batch\_size | 128 | 128 |
| dim\_ff | — | 256 |
| n\_heads | — | 4 |

---

## 3. Implementation

### Frameworks and Libraries

| Component | Library / Version |
|---|---|
| Deep learning | PyTorch |
| Hyperparameter optimisation | Optuna [10] |
| Numerical computation | NumPy |
| Data handling | pandas |
| Visualisation | Matplotlib |
| Progress tracking | tqdm |

### Repository

- **Project repository:** `github.com/robinisaghost/BA_Robin_baseline` (branch: `main`)
- **Model implementations:** `src/ba_baseline/models/lstm.py`, `src/ba_baseline/models/patchtst.py`
- **Data pipeline:** `src/ba_baseline/data/patient_loader.py`, `src/ba_baseline/data/multi_patient_dataset.py`, `src/ba_baseline/data/split.py`
- **Training scripts:** `scripts/train_lstm_60min.py`, `scripts/train_patchtst_60min.py`
- **Metrics:** `src/ba_baseline/metrics/metrics.py`
- **Results:** `reports/results/lstm_60min_summary.json`, `reports/results/patchtst_60min_summary.json`

### External Repositories and References

The following repositories served as implementation references:

| Component | Repository | Role |
|---|---|---|
| PatchTST | `github.com/yuqinie98/PatchTST` (Nie et al. [2]) | Reference implementation; architectural choices (learnable positional embeddings, Pre-LayerNorm, GELU activation) follow this repo |
| LSTM | PyTorch built-in `torch.nn.LSTM` — `github.com/pytorch/pytorch` | Standard library module |
| RevIN | Kim et al. [4]; patterns from `github.com/ts-kim/RevIN` | Normalisation logic follows this reference |

### Key Design Choices

- **Per-patient models:** 36 independent models are trained per architecture, one per patient. This avoids cross-patient interference and reflects the clinical reality that glucose dynamics are highly individual.
- **Fixed random seed:** Reproducibility is ensured by fixing all random seeds.
- **Temporal split:** Data leakage is prevented through strict temporal ordering and per-patient Z-score normalisation on the training split only.
- **Single horizon output:** The LSTM is configured to output a single value (the 60-minute prediction). This contrasts with multi-step output (Objective 2) and makes the time-shift problem maximally visible.

---

## 4. Results

### 4.1 Summary Metrics

| Model | RMSE (mg/dL) | MAE (mg/dL) | Hypo Precision | Hypo Recall | Hypo F1 |
|---|---|---|---|---|---|
| **LSTM MSE** | **36.089** | **27.118** | 0.0104 | 0.0036 | 0.0054 |
| **PatchTST MSE** | **39.271** | **29.001** | 0.0482 | 0.0888 | 0.0609 |

Hypoglycemia event detection uses threshold 70 mg/dL, time tolerance τ = 3 steps (15 min), averaged over 36 patients.

### 4.2 Time-Shift Measurement

A systematic temporal misalignment was measured for both models using cross-correlation-based best-lag detection within a ±12-step (±60-minute) window:

| Model | Mean best-lag (steps) | Mean best-lag (minutes) | Std (steps) |
|---|---|---|---|
| LSTM | 10.22 | 51.1 | 1.84 |
| PatchTST | 11.00 | 55.0 | 0.63 |

Both models predict glucose trajectories that are on average **50–55 minutes shifted** relative to ground truth. This confirms the time-shift phenomenon reported in internal experiments [11]: the model reproduces the correct trend shape but with a delay approaching the full prediction horizon.

### 4.3 Per-Patient Highlights

- RMSE ranges from ~21 mg/dL (patient 85101, stable glucose) to ~59 mg/dL (patient 85204, high variability).
- Hypoglycemia detection is near-zero for LSTM (only patient 85105 achieves F1 > 0). PatchTST detects events in a subset of patients due to the attention mechanism's ability to identify descending patterns.
- The best-lag is notably consistent across patients, especially for PatchTST (std = 0.63 steps = 3.2 minutes), suggesting the shift is a systemic model behaviour rather than a patient-specific artefact.

---

## 5. Interpretation

### 5.1 Time-Shift Artefact

The dominant observation is that both models produce predictions shifted by approximately one full prediction horizon (50–55 min at a 60-min target). This is a well-known pathology of direct pointwise regression on long horizons: MSE minimisation incentivises predicting the recent trend extrapolated forward, which lags behind actual glucose dynamics when they turn or accelerate [11]. The model essentially "predicts the present" rather than the future.

The near-identical shift across patients (especially PatchTST with std = 3 min) indicates this is not noise but a deterministic bias of the MSE loss at this horizon. This directly motivates Objectives 1 (offset-aware losses) and 2 (multi-step supervision).

### 5.2 LSTM vs. PatchTST

LSTM achieves lower RMSE (36.09 vs. 39.27) under standard MSE training. This is consistent with the CGM forecasting literature: recurrent models trained with MSE naturally optimise for the exact pointwise value and can be more data-efficient than Transformer-based models on the relatively short CGM sequences used here [5]. However, PatchTST shows substantially higher hypoglycemia recall (8.9% vs. 0.4%), suggesting the attention mechanism captures longer-range descending glucose patterns more reliably.

### 5.3 Hypoglycemia Detection

The near-zero F1 for LSTM (0.0054) is a direct consequence of class imbalance. Hypoglycemia events (BG < 70 mg/dL) are rare in CGM data, and MSE minimisation incentivises the model to predict the mean glucose trajectory. The model learns stable mid-range predictions that minimise aggregate squared error, systematically ignoring the low-glucose tails. This is the classical imbalanced regression problem [8] and motivates Objective 3 (event-centric evaluation).

### 5.4 Role as Controlled Reference

All design decisions at baseline are held fixed across all extension branches. This ensures any difference in results between Objectives 1–3 and the baseline is attributable solely to the change in loss function or training objective — an ablation-style experimental design [11].

---

## 6. References

[1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/NECO.1997.9.8.1735

[2] Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. *ICLR 2023*. https://openreview.net/forum?id=Jbdc0vTOcol

[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*.

[4] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). Reversible instance normalization for accurate time-series forecasting against distribution shift. *ICLR 2022*. https://openreview.net/forum?id=cGDAkQo1C0p

[5] Lim, B., & Zohren, S. (2020). Time series forecasting with deep learning: A survey. *Philosophical Transactions of the Royal Society A*, 379(2194). https://doi.org/10.1098/rsta.2020.0209

[6] Nemat, H., Khadem, H., Elliott, J., & Benaissa, M. (2024). Data-driven blood glucose level prediction in type 1 diabetes: a comprehensive comparative analysis. *Scientific Reports*, 14(1), 21863. https://doi.org/10.1038/s41598-024-70277-x

[8] Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long short-term memory and graph attention network based approaches. Bachelor Thesis, University of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

[9] Garcia-Tirado, J., Colmegna, P., Villard, O., et al. (2023). Assessment of meal anticipation for improving fully automated insulin delivery in adults with type 1 diabetes. *Diabetes Care*, 46(9), 1652–1658. https://doi.org/10.2337/dc23-0119

[10] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*, 2623–2631. https://doi.org/10.1145/3292500.3330701

[11] Pattern Recognition Group, University of Bern. Glucose Prediction Proposal. Internal unpublished manuscript.
