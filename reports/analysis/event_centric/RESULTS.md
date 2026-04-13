# Objective 3: Event-Centric Evaluation — τ-Sweep

**Branch:** `devBranch-event-centric`
**Approach:** Hypoglycemia event detection evaluated across τ ∈ {1,…,6} steps (5–30 min)
**Models evaluated:** All trained variants (Baseline, Bounded-Lag, Multi-Step)

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

| τ (min) | LSTM base | LSTM BL | LSTM multi | PatchTST base | PatchTST BL | PatchTST multi |
|---|---|---|---|---|---|---|
| 5  | 0.0054 | 0.0060 | 0.0029 | 0.0212 | 0.0311 | 0.0253 |
| 10 | 0.0054 | 0.0126 | 0.0098 | 0.0415 | 0.0574 | 0.0533 |
| 15 | 0.0054 | 0.0141 | 0.0098 | 0.0609 | 0.0665 | 0.0716 |
| 20 | 0.0054 | 0.0141 | 0.0098 | 0.0729 | 0.0837 | 0.0955 |
| 25 | 0.0106 | 0.0141 | 0.0149 | 0.0938 | 0.1170 | 0.1143 |
| 30 | 0.0253 | 0.0226 | 0.0234 | 0.1177 | 0.1389 | 0.1377 |

---

## 3. F2 Results by Model and τ

| τ (min) | LSTM base | LSTM BL | LSTM multi | PatchTST base | PatchTST BL | PatchTST multi |
|---|---|---|---|---|---|---|
| 5  | 0.0042 | 0.0052 | 0.0026 | 0.0251 | 0.0346 | 0.0306 |
| 10 | 0.0042 | 0.0113 | 0.0095 | 0.0491 | 0.0674 | 0.0611 |
| 15 | 0.0042 | 0.0127 | 0.0095 | 0.0742 | 0.0797 | 0.0851 |
| 20 | 0.0042 | 0.0127 | 0.0095 | 0.0899 | 0.1025 | 0.0955 |
| 25 | 0.0084 | 0.0127 | 0.0129 | 0.1155 | 0.1480 | 0.1426 |
| 30 | 0.0192 | 0.0198 | 0.0209 | 0.1467 | 0.1759 | 0.1710 |

---

## 4. Recall Results by Model and τ

| τ (min) | LSTM base | LSTM BL | LSTM multi | PatchTST base | PatchTST BL | PatchTST multi |
|---|---|---|---|---|---|---|
| 5  | 0.0036 | 0.0048 | 0.0024 | 0.0292 | 0.0381 | 0.0386 |
| 10 | 0.0036 | 0.0107 | 0.0095 | 0.0568 | 0.0782 | 0.0720 |
| 15 | 0.0036 | 0.0119 | 0.0095 | 0.0888 | 0.0955 | 0.1032 |
| 20 | 0.0036 | 0.0119 | 0.0095 | 0.1088 | 0.1250 | 0.1153 |
| 25 | 0.0074 | 0.0119 | 0.0122 | 0.1400 | 0.1923 | 0.1850 |
| 30 | 0.0166 | 0.0185 | 0.0201 | 0.1803 | 0.2271 | 0.2196 |

---

## 5. Analysis and Interpretation

### LSTM Baseline — systematic time-shift exposed by τ-sweep

LSTM Baseline F1 is exactly identical from τ=5 to τ=20 (F1=0.0054). This means the event
detections that do occur are all more than 20 minutes away from the true event onset. Recall
does not change across this range either (0.0036 throughout). Only at τ=25 do new matches
appear, and the jump at τ=30 is substantial (F1=0.0253).

This plateau is the clearest quantitative evidence of the time-shift artefact in this study.
LSTM does not fail to detect trends — it detects them too early or too late by a systematic
margin that exceeds 20 minutes. Standard evaluation at τ=15 min understates this problem
by comparing LSTM and PatchTST under a tolerance that already forgives PatchTST's smaller
offset but not LSTM's larger one.

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

### PatchTST scales smoothly across all τ

All three PatchTST variants show monotonically increasing F1 with no plateaus. This reflects
a fundamentally different detection profile: PatchTST's event detections are spread across
the full tolerance range rather than clustered beyond 20 minutes. Even at τ=5 (±5 min),
PatchTST Baseline achieves F1=0.0212 — four times higher than LSTM Baseline at the same
tolerance. This confirms that PatchTST's predictions are temporally closer to true events
across the board.

### PatchTST Bounded-Lag is the strongest variant at larger tolerances

PatchTST Bounded-Lag achieves the highest F1 and F2 at τ=25 and τ=30 (F1=0.1170 and
0.1389; F2=0.1480 and 0.1759). Its recall reaches 0.2271 at τ=30 — the highest of any
model. This is consistent with the per-objective findings: Bounded-Lag improves recall at
the cost of a moderate increase in false positives.

### PatchTST Multi-Step leads at τ=15–20

At τ=15 (F1=0.0716) and τ=20 (F1=0.0955), PatchTST Multi-Step outperforms both PatchTST
Baseline and Bounded-Lag. This suggests that multi-horizon training helps PatchTST detect
events at the standard clinical tolerance window more precisely. At τ=25 and τ=30,
Bounded-Lag overtakes Multi-Step as recall continues to grow.

### Clinical interpretation

For clinical deployment with a 15-minute warning window (τ=3 steps), the ranking is:
PatchTST Multi-Step (F2=0.0851) > PatchTST Bounded-Lag (F2=0.0797) > PatchTST Baseline
(F2=0.0742) >> all LSTM variants (F2 ≤ 0.0127).

If a 30-minute window is acceptable (early warning system), PatchTST Bounded-Lag achieves
recall=0.227 — detecting roughly one in four hypoglycemia events. LSTM remains poor even
at this tolerance (recall=0.016–0.020).

The τ-sweep reveals that the choice of τ is not a minor implementation detail but a
substantive design decision that determines which model appears best. At τ=15 min,
Multi-Step and Bounded-Lag are comparable; at τ=30 min, Bounded-Lag is clearly stronger.
Reporting results at a single fixed τ — as is common in the literature — can obscure these
differences.
