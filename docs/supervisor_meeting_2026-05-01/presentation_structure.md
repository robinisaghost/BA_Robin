# Proposed Presentation Structure

**Thesis:** Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction  
**Student:** Robin van den Hoek  
**Supervisor:** PD Dr. Kaspar Riesen  
**Duration:** ~30 min + Q&A  
**Format:** LaTeX/Beamer (PRG Seminar guidelines)

---

*Use this document during the meeting to mark which slides to keep, shorten, or drop.*  
*Legend: `[keep]` `[shorten]` `[drop]` `[discuss]`*

---

## Part 1 — Motivation & Context (~6 min, ~4 slides)

**Slide 1 — Title**
- Thesis title, name, supervisor, date
- `[ ]`

**Slide 2 — Clinical motivation**
- CGM devices, Type 1 diabetes management, the danger of hypoglycemia
- What does a 55-minute prediction delay mean for a patient trying to eat sugar in time?
- Key message: timing matters as much as accuracy
- `[ ]`

**Slide 3 — The time-shift problem**
- Figure: example glucose curve with prediction overlaid showing the lag
- Introduce lag_rmse as the metric that isolates the shift component from total RMSE
- Key message: MSE-trained models reproduce the right shape, but always too late
- `[ ]`

**Slide 4 — Related work & positioning**
- Hüni (2023): LSTM vs. GAT for hypoglycemia prediction at University of Bern
- This thesis: same dataset, same evaluation protocol — extends Hüni with three systematic mitigations
- Key message: builds on established baseline, controlled ablation design (one factor at a time)
- `[ ]`

---

## Part 2 — Experimental Setup & Baseline (~5 min, ~2 slides)

**Slide 5 — Experimental setup**
- 36 patients, 5-min intervals, lookback=24 steps (2h), horizon h=11 (60 min)
- Train/Val/Test: 60/20/20 temporal split per patient, 36 patient-specific models
- Two architectures: LSTM and PatchTST
- Hyperparameter tuning: Optuna on patient 85202, all objectives use same HP
- Metrics: RMSE (60-min accuracy), lag_rmse (shift diagnosis), F1/F2 (hypoglycemia events)
- `[ ]`

**Slide 6 — Baseline results**
- Table: LSTM RMSE=36.09 / PatchTST RMSE=39.27; both F1 near zero
- Mean temporal shift: LSTM 10.2 steps (51 min), PatchTST 11.0 steps (55 min)
- Key message: MSE optimisation produces accurate shapes but catastrophic event detection — motivates all three objectives
- `[ ]`

---

## Part 3 — Objective 1: Offset-Aware Losses (~6 min, ~3 slides)

**Slide 7 — Objective 1a: Bounded-Lag loss**
- Formula: `L = min_{Δ ∈ [−D,D]} MSE(ŷ, y_Δ)`, D=3 steps (±15 min)
- Rationale: within D steps, alignment is free — gradients flow through best-matching offset
- Results: RMSE negligible change (<0.05); PatchTST lag_rmse=16.70 (best of Obj.1); slight F1 improvement for LSTM
- `[ ]`

**Slide 8 — Objective 1b: Soft-DTW loss**
- Concept: differentiable DTW with smoothing γ=1.0; allows elastic alignment along full path
- Results: RMSE regression (+1.1 LSTM, +3.96 PatchTST); lag_rmse worse than Bounded-Lag
- Why it underperforms: DTW deprioritises exact amplitude → penalised by RMSE; 50-min shift spans nearly the full prediction horizon, unlike the speech recognition setting DTW was designed for
- `[ ]`

**Slide 9 — Objective 1 conclusion**
- Summary table for all four model-loss combinations
- Key message: D=3 covers only ~30% of the actual shift; loss relaxation reduces training penalty but cannot supply the structural information needed to predict earlier
- Transition: → need stronger supervision signal (Objective 2)
- `[ ]`

---

## Part 4 — Objective 2: Multi-Step Forecasting (~5 min, ~2 slides)

**Slide 10 — Multi-step forecasting concept**
- DIRMO strategy: model outputs all 12 steps simultaneously, loss averaged over full trajectory
- Hypothesis: predicting intermediate steps keeps the model "in phase" — temporal drift is penalised at every step, not just h=11
- Evaluation still at h=11 for fair comparison with baseline
- `[ ]`

**Slide 11 — Multi-step results**
- Table: LSTM RMSE=35.59 (−0.50), lag_rmse=20.74; PatchTST RMSE=39.05 (−0.22), lag_rmse=14.53
- Key message: **lowest lag_rmse of any tested method** — but shift penalty remains 40–63% of RMSE; structural drift not fully eliminated
- Best approach for raw glucose forecasting accuracy
- `[ ]`

---

## Part 5 — Objective 3: Event-Centric (~6 min, ~3 slides)

**Slide 12 — τ-sweep: quantifying the clinical cost of the shift**
- Plot: F1 vs. tolerance τ (0–12 steps) for all forecasting models
- Key message: all forecasting models require τ>5 steps (25 min) for non-trivial recall; at τ=3 (15 min, clinically appropriate), forecast-derived detection fails
- This provides a quantitative argument for the direct classifier approach
- `[ ]`

**Slide 13 — Direct binary event classifier**
- Architecture: same LSTM/PatchTST with sigmoid output head, BCE loss
- Label: 1 if any value in y_{t+0}…y_{t+11} < 70 mg/dL
- Results: LSTM F1=0.334 (**62× improvement** over baseline F1=0.005); Recall=0.693
- PatchTST F1=0.095 (1.6× improvement); very high false positive rate
- `[ ]`

**Slide 14 — Objective 3 conclusion: architecture inversion**
- LSTM: worse at forecasting (RMSE 36.09 vs 39.27) but 3.5× better at event classification (F1 0.334 vs 0.095)
- Interpretation: recurrent state tracking suits binary event prediction; PatchTST's attention fires on any descending trend
- Key message: MSE training and event detection are fundamentally misaligned objectives → must be separated
- `[ ]`

---

## Part 6 — Synthesis & Close (~4 min, ~3 slides)

**Slide 15 — Overall comparison & recommendation**
- Full comparison table across all methods and metrics (RMSE, lag_rmse, Hypo-F1)
- Recommendation: Multi-step MSE for glucose trajectory; LSTM binary classifier for hypoglycemia warning
- Consistent with Hüni (2023): treat forecasting and event prediction as separate tasks
- `[ ]`

**Slide 16 — Limitations & future work**
- D=3 window in Bounded-Lag too narrow for a 50-min shift; larger D or learned window could help
- Class imbalance: LSTM precision still only 0.237 — threshold optimisation or cost-sensitive training not yet explored
- Optional Objective 4: post-hoc shift correction (constant k* subtracted at test time)
- Per-patient variability large (RMSE std ~7–8 mg/dL) — personalisation unexplored
- `[ ]`

**Slide 17 — Discussion questions**
- Q1: The bounded-lag window D=3 covers only 30% of the observed shift. Would a larger window (e.g., D=6 or D=11) be worth exploring, and how would you set the boundary?
- Q2: The direct LSTM event classifier achieves Recall=0.69 but Precision=0.24 — 3 out of 4 alarms are false. Is this acceptable for a clinical alert system, and how should the threshold be calibrated?
- `[ ]`

**Slide 18 — References**
- Hüni (2023), Cuturi & Blondel (2017), Sakoe & Chiba (1978), relevant LSTM/PatchTST papers
- `[ ]`

---

## Time budget (rough guide)

| Part | Slides | Time |
|------|--------|------|
| Motivation & Context | 4 | ~6 min |
| Setup & Baseline | 2 | ~5 min |
| Objective 1 | 3 | ~6 min |
| Objective 2 | 2 | ~5 min |
| Objective 3 | 3 | ~6 min |
| Synthesis & Close | 3 | ~4 min |
| **Total** | **17** | **~30 min** |

---

## Notes from meeting

*(Freigelassen für Notizen während des Gesprächs)*
