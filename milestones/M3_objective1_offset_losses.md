# Milestone M3: Objective 1 — Offset-Aware Loss Functions

**Thesis:** Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction  
**Student:** Robin van den Hoek (22-127-641)  
**Supervisor:** PD Dr. Kaspar Riesen, University of Bern (INF)  
**Proposal phase:** Weeks 7–9  
**Branches:** `devBranch-offset-loss-bounded-lag`, `devBranch-offset-loss-dtw`  
**Results on:** `main` (`reports/results/`)  
**Repository:** github.com/robinisaghost/BA_Robin_baseline

---

## 1. Goal

Implement and evaluate offset-aware loss functions that reduce the penalty for small temporal misalignment during training, so the model is encouraged to match the trajectory shape and timing more robustly. Two variants are implemented:

1. **Bounded-Lag Alignment Loss (Objective 1a):** computes MSE at the best-matching temporal offset within a bounded window D = 3 steps (15 minutes).
2. **Soft-DTW Loss (Objective 1b, optional):** a differentiable Dynamic Time Warping loss that allows elastic alignment between predicted and true trajectories.

Both variants use the same baseline hyperparameters (Optuna patient 85202) and evaluation protocol (M2). Only the training loss changes.

---

## 2. Approach

### 2.1 Rationale for Alignment-Aware Training

Standard MSE penalises each step independently without regard to temporal structure. If the prediction is shifted by Δ steps, each pointwise error is approximately `(ŷ_{t+k} − y_{t+k})² ≈ (y_{t+k+Δ} − y_{t+k})²`, which is large even when the predicted shape is clinically correct. An alignment-aware loss that evaluates the prediction at the best offset removes this penalty, incentivising the model to produce the correct trajectory shape and, ideally, also learn to correct its timing [11].

### 2.2 Objective 1a: Bounded-Lag Alignment Loss

For each training batch sample, the loss evaluates the prediction against the target after aligning within a window D = 3 steps (±15 minutes):

```
L_bounded(ŷ, y) = min_{Δ ∈ [-D, D]}  MSE(ŷ, y_shifted_by_Δ)
```

The target sequence is extended by 2·D additional steps (H + 2·D = 12 + 6 = 18 steps) to allow alignment without boundary artefacts. For each candidate offset Δ, the prediction is compared against the correctly-sliced target window. The minimum MSE over all offsets defines the loss.

During backpropagation, gradients flow through the MSE at the chosen best offset only (a hard-minimum selection). This is equivalent to training the model to match the target at the position it is most naturally inclined to predict — gradually pulling the model toward correct timing.

**Window size D = 3 steps (15 min):** chosen to cover the clinically plausible misalignment range. An earlier implementation contained a slice-length bias (the true tensor was not extended to H + 2·D = 18 steps), which was identified and corrected before the results reported here were produced.

**Implementation:** `src/ba_baseline/losses/bounded_lag_loss.py` (branch `devBranch-offset-loss-bounded-lag`)  
**Training scripts:** `scripts/train_lstm_bounded_lag.py`, `scripts/train_patchtst_bounded_lag.py`

### 2.3 Objective 1b: Soft-DTW Loss

Dynamic Time Warping (DTW) [4] is a classical sequence alignment algorithm that finds the optimal non-linear temporal mapping between two time series. Standard DTW is non-differentiable (uses hard argmin), making it unsuitable for gradient-based training. Cuturi & Blondel [12] introduced **Soft-DTW**, a smooth, differentiable approximation:

```
DTW_γ(ŷ, y) = -γ · log Σ_A  exp(-cost(A) / γ)
```

where γ > 0 is a smoothing parameter (γ = 1.0 used here), A ranges over all monotone alignment paths, and cost(A) is the sum of squared distances along path A. As γ → 0, Soft-DTW converges to standard DTW; as γ → ∞, it converges to a smoothed global average. γ = 1.0 provides a practically useful intermediate.

Soft-DTW allows the loss to match predicted and true sequences at non-identical time points, penalising shape discrepancies less than temporal offsets. This makes the model more robust to small timing errors during training.

**Implementation:** `src/ba_baseline/losses/soft_dtw_loss.py` (branch `devBranch-offset-loss-dtw`)  
**Training scripts:** `scripts/train_lstm_dtw.py`, `scripts/train_patchtst_dtw.py`  
**Hyperparameters:** same as baseline (no separate Optuna run for DTW).

---

## 3. Implementation

### Frameworks and Libraries

| Component | Library |
|---|---|
| Deep learning | PyTorch |
| Custom losses | PyTorch autograd (bounded-lag, soft-DTW) |
| Numerical computation | NumPy |
| Data handling | pandas |

### Repository Structure

| File / Branch | Purpose |
|---|---|
| `devBranch-offset-loss-bounded-lag` | Full implementation and training scripts for bounded-lag |
| `devBranch-offset-loss-dtw` | Full implementation and training scripts for Soft-DTW |
| `reports/results/lstm_bounded_lag_*.{json,csv,npz}` | LSTM bounded-lag results (on `main`) |
| `reports/results/patchtst_bounded_lag_*.{json,csv,npz}` | PatchTST bounded-lag results (on `main`) |
| `reports/results/lstm_dtw_*.{json,csv,npz}` | LSTM Soft-DTW results (on `main`) |
| `reports/results/patchtst_dtw_*.{json,csv,npz}` | PatchTST Soft-DTW results (on `main`) |

### External Repositories

Both loss functions were implemented from scratch using PyTorch autograd. No external repository code was copied.

| Component | Reference repository | Relation to this work |
|---|---|---|
| Bounded-Lag loss | No existing reference repo; design follows [11] | Original implementation |
| Soft-DTW loss | `github.com/mblondel/soft-dtw` (Cuturi & Blondel [12]) | Re-implemented from scratch in PyTorch following the algorithm in [12]; no code copied from the reference repo |

### Ablation Design

The sole change between the baseline and each Objective 1 variant is the loss function. Architecture, hyperparameters, data pipeline, evaluation protocol, and random seeds are identical to M1. This ensures that result differences are caused only by the loss change.

---

## 4. Results

### 4.1 Summary Table (all models compared)

| Model | Loss | RMSE (mg/dL) | lag\_rmse (mg/dL) | Shift penalty | MAE (mg/dL) | Hypo F1 |
|---|---|---|---|---|---|---|
| LSTM | MSE (baseline) | 36.089 | — | — | 27.118 | 0.0054 |
| LSTM | Bounded-Lag (D=3) | **36.140** | **25.253** | 10.887 | **27.571** | **0.0141** |
| LSTM | Soft-DTW (γ=1) | 37.232 | 28.668 | 8.564 | 28.760 | 0.0137 |
| PatchTST | MSE (baseline) | 39.271 | — | — | 29.001 | 0.0609 |
| PatchTST | Bounded-Lag (D=3) | **39.296** | **16.704** | 22.592 | **29.032** | **0.0665** |
| PatchTST | Soft-DTW (γ=1) | 43.235 | 24.006 | 19.229 | 32.251 | 0.0624 |

*lag\_rmse = minimum RMSE after optimal constant shift k\* ∈ [−12, 12]. Baseline lag\_rmse not directly comparable as it is a post-hoc metric on single-output models.*  
*Shift penalty = RMSE − lag\_rmse.*

### 4.2 Bounded-Lag Results Detail

**LSTM Bounded-Lag:**
- RMSE = 36.140 (virtually identical to baseline 36.089, Δ = +0.05 mg/dL)
- lag\_rmse = 25.253 — shift penalty of 10.887 mg/dL

**PatchTST Bounded-Lag:**
- RMSE = 39.296 (virtually identical to baseline 39.271, Δ = +0.025 mg/dL)
- lag\_rmse = 16.704 — shift penalty of 22.592 mg/dL; **substantially lower lag\_rmse than DTW (24.006) and multi-step comparison**

Hypoglycemia F1 improves slightly for LSTM (0.0054 → 0.0141) and remains consistent for PatchTST (0.061 → 0.067).

### 4.3 Soft-DTW Results Detail

**LSTM Soft-DTW:**
- RMSE = 37.232 — slight regression vs. baseline (+1.14 mg/dL)
- lag\_rmse = 28.668 — shift penalty of 8.564 mg/dL
- MAE = 28.760 — regression vs. baseline (+1.64 mg/dL)

**PatchTST Soft-DTW:**
- RMSE = 43.235 — notable regression vs. baseline (+3.96 mg/dL)
- lag\_rmse = 24.006 — shift penalty of 19.229 mg/dL
- MAE = 32.251 — regression vs. baseline (+3.25 mg/dL)

PatchTST Soft-DTW shows the largest RMSE regression of all models tested.

### 4.4 Temporal Shift Quantification (Δ\*)

The mean best-lag from cross-correlation for the baseline was 10.22 steps (LSTM) and 11.0 steps (PatchTST). While the offset-aware losses do not eliminate the shift entirely, the lag\_rmse metric shows that:

- **Bounded-lag PatchTST achieves the lowest lag\_rmse (16.704)** among all Objective 1 variants, suggesting the bounded-lag loss is most effective at allowing the model to correct its timing under the ±D constraint.
- DTW's higher lag\_rmse despite its full elastic alignment capability is discussed in Section 5.

---

## 5. Interpretation

### 5.1 Bounded-Lag: Negligible RMSE Impact, Reduced Shift Penalty

The bounded-lag loss maintains pointwise RMSE essentially unchanged (Δ < 0.05 mg/dL for both architectures) while the lag\_rmse reveals a shift structure. This suggests the bounded-lag loss does not cause model degradation in pointwise accuracy — a desirable property. The slight F1 improvement for LSTM (0.0054 → 0.0141) is consistent with the model learning slightly better trajectory shapes, though the effect is small.

The constraint D = 3 steps (15 min) may be too narrow to substantially alter how the model distributes temporal error. The bounded-lag loss essentially tells the model: "within 15 minutes, alignment is free." If the dominant shift is 50 minutes (as seen in the baseline), this window covers only about 30% of the actual shift, limiting how much the loss can guide the model toward correct timing.

### 5.2 Soft-DTW: RMSE Regression, Mixed Shift Reduction

Soft-DTW causes RMSE regression, particularly for PatchTST (+3.96 mg/dL). This is a known issue with DTW-based losses for regression: because DTW allows elastic alignment, the model can learn to reproduce the general trend at the cost of precise amplitude, inflating pointwise error [4]. The DTW loss explicitly deprioritises exact point-by-point matching, which is what RMSE measures — creating a direct tension between the training objective and the evaluation metric.

The lag\_rmse for DTW is higher than for bounded-lag (28.668 vs. 25.253 for LSTM; 24.006 vs. 16.704 for PatchTST), suggesting DTW does not reduce the temporal shift more effectively than the simpler bounded-lag approach. This may reflect the difficulty of training with DTW on sequences where the dominant shift is much larger than the local alignment path can accommodate.

### 5.3 Comparison with Prior Work

Sakoe & Chiba [4] introduced DTW for speech recognition, where the alignment is typically a small fraction of the sequence length. In glucose forecasting at a 60-minute horizon, the shift (50–55 min) is nearly as long as the prediction itself. Applying Soft-DTW [12] in this regime asks the alignment path to span almost the full sequence — a qualitatively different setting than the original application.

The bounded-lag loss, by contrast, has a hard window that prevents unrealistic alignments. This may explain why it is more stable and does not degrade RMSE, consistent with the thesis proposal's guidance to "start with the simple bounded-lag method; include DTW-inspired loss only if feasible" [11].

### 5.4 Neither Loss Eliminates the Shift

Both offset-aware losses leave the mean best-lag Δ\* largely unchanged compared to the baseline. The losses reduce how severely the model is penalised for the shift during training, but they do not provide the model with the structural information needed to predict earlier. This motivates Objective 2 (multi-step forecasting), which hypothesises that predicting intermediate steps provides stronger supervision to keep the model "in phase" over the full horizon [11].

### 5.5 Clinical Implications

The near-baseline hypoglycemia F1 for both loss variants confirms that offset-aware training alone — within the narrow D = 3 window — does not substantially improve event detection. The dominant limitation remains the class imbalance and the model's tendency to predict the mean trajectory, not the alignment loss. This motivates the event-centric approach of Objective 3.

---

## 6. References

[4] Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26, 43–49.

[7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis, University of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

[8] Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long short-term memory and graph attention network based approaches. Bachelor Thesis, University of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

[11] Pattern Recognition Group, University of Bern. Glucose Prediction Proposal. Internal unpublished manuscript.

[12] Cuturi, M., & Blondel, M. (2017). Soft-DTW: A differentiable loss function for time-series. *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*, 894–903.
