# Objective 3: Event-Centric Evaluation

**Branch:** `devBranch-event-centric`
**Part 1:** τ-sweep of forecast-derived event detection across τ ∈ {1,…,6} steps (5–30 min)
**Part 2:** Direct binary event classifiers (LSTM Event, PatchTST Event)
**Models evaluated:** All trained variants (Baseline, Bounded-Lag, DTW, Multi-Step) plus direct classifiers

---

## 1. Method

All models output a predicted glucose trajectory. An event is defined as a threshold crossing
from above to below 70 mg/dL (hypoglycemia onset). A predicted crossing is counted as a True
Positive (TP) if it falls within ±τ steps of a true crossing. False Positives (FP) are
predicted crossings with no matching true event; False Negatives (FN) are true crossings not
matched by any prediction.

Rather than fixing τ = 3 (15 min) as in the main evaluation, this analysis sweeps
τ ∈ {1, 2, 3, 4, 5, 6} steps = {5, 10, 15, 20, 25, 30} minutes. This reveals how robust
each model's event timing is: a model that only scores well at large τ has a systematic
time-shift in its detections; a model that scores well even at τ = 1 detects events close
to their true onset.

F2 (β=2) is reported alongside F1. F2 weights recall twice as heavily as precision,
reflecting the clinical priority of not missing hypoglycemic events [2].

---

## 2. F1 Results by Model and τ

| τ (min) | LSTM base | LSTM BL | LSTM DTW | LSTM multi | PatchTST base | PatchTST BL | PatchTST DTW | PatchTST multi |
|---|---|---|---|---|---|---|---|---|
| 5  | 0.0054 | 0.0060 | 0.0017 | 0.0029 | 0.0212 | 0.0311 | 0.0435 | 0.0253 |
| 10 | 0.0054 | 0.0126 | 0.0051 | 0.0098 | 0.0415 | 0.0574 | 0.0577 | 0.0533 |
| 15 | 0.0054 | 0.0141 | 0.0137 | 0.0098 | 0.0609 | 0.0665 | 0.0624 | 0.0716 |
| 20 | 0.0054 | 0.0141 | 0.0137 | 0.0098 | 0.0729 | 0.0837 | 0.0977 | 0.0801 |
| 25 | 0.0106 | 0.0141 | 0.0137 | 0.0149 | 0.0938 | 0.1170 | 0.1140 | 0.1143 |
| 30 | 0.0253 | 0.0226 | 0.0218 | 0.0234 | 0.1177 | 0.1389 | 0.1247 | 0.1377 |

---

## 3. F2 Results by Model and τ

| τ (min) | LSTM base | LSTM BL | LSTM DTW | LSTM multi | PatchTST base | PatchTST BL | PatchTST DTW | PatchTST multi |
|---|---|---|---|---|---|---|---|---|
| 5  | 0.0042 | 0.0052 | 0.0014 | 0.0026 | 0.0251 | 0.0346 | 0.0568 | 0.0306 |
| 10 | 0.0042 | 0.0113 | 0.0041 | 0.0095 | 0.0491 | 0.0674 | 0.0757 | 0.0611 |
| 15 | 0.0042 | 0.0127 | 0.0102 | 0.0095 | 0.0742 | 0.0797 | 0.0838 | 0.0851 |
| 20 | 0.0042 | 0.0127 | 0.0102 | 0.0095 | 0.0899 | 0.1025 | 0.1235 | 0.0955 |
| 25 | 0.0084 | 0.0127 | 0.0102 | 0.0129 | 0.1155 | 0.1480 | 0.1477 | 0.1426 |
| 30 | 0.0192 | 0.0198 | 0.0173 | 0.0209 | 0.1467 | 0.1759 | 0.1627 | 0.1710 |

---

## 4. Recall Results by Model and τ

| τ (min) | LSTM base | LSTM BL | LSTM DTW | LSTM multi | PatchTST base | PatchTST BL | PatchTST DTW | PatchTST multi |
|---|---|---|---|---|---|---|---|---|
| 5  | 0.0036 | 0.0048 | 0.0012 | 0.0024 | 0.0292 | 0.0381 | 0.0862 | 0.0386 |
| 10 | 0.0036 | 0.0107 | 0.0036 | 0.0095 | 0.0568 | 0.0782 | 0.1131 | 0.0720 |
| 15 | 0.0036 | 0.0119 | 0.0088 | 0.0095 | 0.0888 | 0.0955 | 0.1288 | 0.1032 |
| 20 | 0.0036 | 0.0119 | 0.0088 | 0.0095 | 0.1088 | 0.1250 | 0.1761 | 0.1153 |
| 25 | 0.0074 | 0.0119 | 0.0088 | 0.0122 | 0.1400 | 0.1923 | 0.2231 | 0.1850 |
| 30 | 0.0166 | 0.0185 | 0.0154 | 0.0201 | 0.1803 | 0.2271 | 0.2453 | 0.2196 |

---

## 5. Analysis and Interpretation

### LSTM Baseline: systematic time-shift exposed by τ-sweep

LSTM Baseline F1 is exactly identical from τ=5 to τ=20 (F1=0.0054). This means the event
detections that do occur are all more than 20 minutes away from the true event onset. Recall
does not change across this range either (0.0036 throughout). Only at τ=25 do new matches
appear, and the jump at τ=30 is substantial (F1=0.0253).

This plateau is the clearest quantitative evidence of the time-shift artefact in this study.
LSTM does not fail to detect trends, it detects them too early or too late by a systematic
margin that exceeds 20 minutes. Standard evaluation at τ=15 min understates this problem
by comparing LSTM and PatchTST under a tolerance that already forgives PatchTST's smaller
offset but not LSTM's larger one.

### LSTM DTW: worse than Baseline at tight tolerances, plateau persists

LSTM DTW achieves F1=0.0017 at τ=5 — the lowest of any model and below even LSTM Baseline
(0.0054). Recall at τ=5 is only 0.0012. At τ=10, F1 climbs to 0.0051 but remains below
Baseline. From τ=15 to τ=25, F1 is flat at 0.0137 — the same plateau pattern as LSTM
Baseline, but shifted to a different (higher) tolerance threshold.

This result is consistent with the lag_rmse finding: DTW training does not reduce LSTM's
time-shift artefact and in fact worsens event detection timing. The DTW loss is invariant
to temporal distortions — the model learns that shifted predictions are acceptable because
the alignment path can warp them to match true trajectories. This removes the gradient
signal that would otherwise penalise temporal offset, allowing LSTM to drift further from
true event timing than with pointwise MSE.

### Bounded-Lag breaks the LSTM plateau

LSTM Bounded-Lag is the first LSTM variant to show improvement within the clinically
relevant window: F1 jumps from 0.0060 at τ=5 to 0.0126 at τ=10, then stabilises at 0.0141
from τ=15 to τ=25. This indicates that Bounded-Lag shifts some of LSTM's event detections
from beyond 20 minutes to within 10 minutes of true events — a meaningful improvement in
detection timing.

However, at τ=30, LSTM Bounded-Lag (F1=0.0226) is marginally below LSTM Baseline
(F1=0.0253). The Bounded-Lag loss appears to trade some of the very-wide-tolerance detections
for better-timed but still imperfect ones. At the clinically most relevant tolerances
(τ=10–20 min), Bounded-Lag consistently outperforms Baseline.

### Multi-Step shifts detection timing but does not improve τ=5

LSTM Multi-Step has lower F1 than Baseline at τ=5 (0.0029 vs 0.0054). This is not a
contradiction of the monotonicity property (each model's F1 is non-decreasing with τ), but
a difference in the distribution of detection offsets between the two models. LSTM Baseline
has a few detections that happen to fall within ±5 minutes of true events (giving the τ=5
score), while the remainder are far beyond 25 minutes.

LSTM Multi-Step has fewer detections within ±5 min but more in the 5–10 minute window:
recall jumps from 0.0024 to 0.0095 between τ=5 and τ=10, while Baseline stays flat. This
suggests that Multi-Step training shifts the event detection distribution: detections are
more consistently spaced 5–10 minutes from true events, rather than split between very close
and very far. At τ=25 and τ=30, Multi-Step and Bounded-Lag reach similar performance.

### PatchTST scales across all τ

All four PatchTST variants show monotonically increasing F1 with no plateaus. This reflects
a fundamentally different detection profile: PatchTST's event detections are spread across
the full tolerance range rather than clustered beyond 20 minutes. Even at τ=5 (±5 min),
PatchTST Baseline achieves F1=0.0212, four times higher than LSTM Baseline at the same
tolerance. This confirms that PatchTST's predictions are temporally closer to true events
across the board.

### PatchTST DTW: highest recall at tight tolerances, high false positive rate

PatchTST DTW shows a unique profile in the τ-sweep. At τ=5, it achieves recall=0.0862 —
the highest of any model at this tolerance and more than twice the next best (PatchTST
Multi-Step: 0.0386). At τ=30, recall reaches 0.2453, again the highest of any model,
exceeding PatchTST Bounded-Lag (0.2271).

However, the high recall comes with low precision across all τ values. At τ=5, precision
is only 0.0429, resulting in F1=0.0435. While this is the highest F1 at τ=5, the F2
(0.0568) is also the highest — suggesting PatchTST DTW generates many threshold crossings,
of which a relatively large number happen to fall near true events but many are false
positives. This pattern is consistent with DTW shape alignment encouraging the model to
reproduce the shape of glucose dips more aggressively, producing more crossings at the cost
of temporal and amplitude precision.

As τ increases from 5 to 20, PatchTST DTW's recall grows steeply (0.0862 → 0.1761)
while Bounded-Lag's grows more moderately (0.0381 → 0.1250). By τ=20, PatchTST DTW
overtakes Bounded-Lag in F1 (0.0977 vs 0.0837) and F2 (0.1235 vs 0.1025). At τ=30,
PatchTST DTW's F2=0.1627 falls below Bounded-Lag (0.1759) because the persistent false
positive cost outweighs the recall advantage when equally weighting both components.

For recall-prioritised clinical settings (early warning with high sensitivity), PatchTST DTW
is the strongest forecast-derived model. For balanced detection (F2 at τ=15–30), Bounded-Lag
is preferred.

### PatchTST Bounded-Lag is the strongest variant at larger tolerances (F2)

PatchTST Bounded-Lag achieves the highest F2 at τ=25 and τ=30 (F2=0.1480 and 0.1759).
Its recall reaches 0.2271 at τ=30, second only to PatchTST DTW. This is consistent with
the per-objective findings: Bounded-Lag improves recall at the cost of a moderate increase
in false positives, but the balance is better controlled than DTW.

### PatchTST Multi-Step leads at τ=15

At τ=15, PatchTST Multi-Step (F1=0.0716, F2=0.0851) outperforms all other models. This
suggests that multi-horizon training helps PatchTST detect events at the standard clinical
tolerance window most precisely. At τ=20, PatchTST DTW overtakes Multi-Step in F1 and F2.
At τ=25 and τ=30, Bounded-Lag is strongest in F2.

### Summary

For clinical deployment with a 15-minute warning window (τ=3 steps), the F2 ranking is:
PatchTST Multi-Step (0.0851) > PatchTST DTW (0.0838) > PatchTST Bounded-Lag (0.0797) >
PatchTST Baseline (0.0742) >> all LSTM variants (F2 ≤ 0.0127).

If maximum recall is the priority (a recall-first early warning system), PatchTST DTW
achieves the highest recall across all τ values — 0.0862 at τ=5 and 0.2453 at τ=30.
If F2 (recall-weighted balance) is the priority, PatchTST Bounded-Lag leads at τ≥25.

The τ-sweep reveals that the choice of τ is not a minor implementation detail but a
substantive design decision that determines which model appears best. At τ=15 min,
Multi-Step and DTW are comparable; at τ=30 min, Bounded-Lag leads in F2. DTW's advantage
is purely in raw recall, at the cost of a persistently higher false positive rate.
The LSTM results confirm that DTW training does not improve and worsens the time-shift
artefact relative to MSE.

---

## 6. Binary Event Classifier: Method

The binary event classifier is a direct alternative to forecast-derived event detection.
Instead of predicting the glucose trajectory and then checking for threshold crossings,
the classifier is trained to answer directly: will glucose fall below 70 mg/dL at any
point in the next 60 minutes?

For each input window of 24 steps (120 minutes), the label is 1 if the minimum of the
next 12 steps (60 minutes) falls below 70 mg/dL, and 0 otherwise. BCEWithLogitsLoss
with a patient-specific class weight (n_neg / n_pos) is used to handle the rarity of
hypoglycemia events.

The same backbone is used as in the baseline: LSTMForecaster with horizon=1 for LSTM
Event, and PatchTST with horizon=1 for PatchTST Event. The single output is treated as
a binary logit. Baseline hyperparameters from Optuna tuning on patient 85202 are used
without modification, consistent with all other objectives. Patients with no positive
events in the training split are skipped.

---

## 7. Binary Event Classifier: Results

### 7.1 Mean Results Across Patients

| Model | Precision | Recall | F1 | F2 |
|---|---|---|---|---|
| LSTM Event | 0.2373 | 0.6931 | 0.3336 | 0.4626 |
| PatchTST Event | 0.0544 | 0.6150 | 0.0952 | 0.1785 |

Both classifiers were trained and evaluated on 36 patients.

### 7.2 Comparison with Forecast-Derived Detection at τ=30 min

τ=30 min is the most forgiving tolerance in the τ-sweep. The comparison at τ=30
therefore represents the upper bound of forecast-derived detection performance.

| Model | F1 | F2 |
|---|---|---|
| LSTM Event | 0.3336 | 0.4626 |
| PatchTST Bounded-Lag (τ=30) | 0.1389 | 0.1759 |
| PatchTST Multi-Step (τ=30) | 0.1377 | 0.1710 |
| PatchTST DTW (τ=30) | 0.1247 | 0.1627 |
| PatchTST Event | 0.0952 | 0.1785 |
| PatchTST Baseline (τ=30) | 0.1177 | 0.1467 |
| LSTM Bounded-Lag (τ=30) | 0.0226 | 0.0198 |
| LSTM Baseline (τ=30) | 0.0253 | 0.0192 |

---

## 8. Binary Event Classifier: Interpretation

### LSTM Event outperforms all forecast-derived models

LSTM Event achieves F2=0.4626. The best forecast-derived model at the most forgiving
tolerance (PatchTST Bounded-Lag at τ=30) achieves F2=0.1759. LSTM Event is 2.6 times
higher in F2. The same ordering holds for F1 and recall.

This result shows that directly training a model to detect hypoglycemia events produces
substantially better results than deriving event detections from a glucose trajectory
forecast. The forecast models are trained to minimise mean squared error of the predicted
trajectory. This objective does not directly reward correct detection of threshold
crossings and does not penalise missed or false events.

### PatchTST Event underperforms relative to LSTM Event

PatchTST Event achieves F2=0.1785. This is comparable to the best forecast-derived
models at τ=30, which means direct binary training gives PatchTST no clear advantage
over its own forecast-derived detection. The main reason is low precision: PatchTST Event
achieves a mean precision of 0.054, meaning approximately 1 in 18 positive predictions
is a true positive. Per-patient inspection shows that PatchTST Event generates between
800 and 1400 false positive predictions per patient, compared to 100 to 400 for LSTM Event.

### Why PatchTST Event has lower precision

PatchTST uses Reversible Instance Normalization (RevIN), which normalizes each input
window by the mean and standard deviation of that specific window before the model
processes it [1]. After RevIN normalization, two windows with the same shape but
different absolute glucose levels are represented identically. For example, a window
centered at 90 mg/dL and a window centered at 150 mg/dL produce the same normalized
input if they have the same internal shape.

For binary classification of an absolute threshold (70 mg/dL), the absolute glucose
level of the input window is informative. A patient currently at 90 mg/dL is much closer
to the hypoglycemia threshold than a patient at 150 mg/dL. After RevIN, the model cannot
use this difference. It can only use the shape and trend within the window.

LSTM Event normalizes inputs using the mean and standard deviation of the training set
for each patient, which are fixed across all windows. This means the normalized value of
70 mg/dL is always the same for a given patient, and the model can learn that normalized
values below a certain level indicate proximity to the threshold. The absolute level
information is preserved in the normalized representation.

### The reversal relative to the τ-sweep

In the τ-sweep, PatchTST outperformed LSTM across all models and all tolerance values.
In direct binary classification, LSTM outperforms PatchTST by a large margin. This
shows that the same architectural choices that make PatchTST better for glucose
trajectory forecasting make it less suited for direct binary classification of an
absolute threshold. RevIN is beneficial for forecasting because it reduces sensitivity
to patient-level distribution shift [1]. For binary threshold classification, it removes
the information that matters most.

---

## References

[1] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2022). Reversible
    instance normalization for accurate time-series forecasting against distribution
    shift. In The Tenth International Conference on Learning Representations (ICLR 2022).
    https://openreview.net/forum?id=cGDAkQo1C0p
