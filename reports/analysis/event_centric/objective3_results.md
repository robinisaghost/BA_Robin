# Objective 3: Event-Centric Evaluation — Results

This document summarises all models trained and evaluated for Objective 3.
The goal of Objective 3 is to directly predict hypoglycemia events
(glucose dropping below 70 mg/dL) rather than forecasting the full glucose
trajectory and deriving events after the fact.

All models use the same lookback window of **24 steps (2 hours)** as input.
Evaluation is on the held-out **test split** (last 20 % of each patient's data).

---

## Naive Baselines

These are simple rule-based predictors that require no training.
They serve as a sanity check: a trained model should outperform them,
otherwise the model has not learned anything useful.

The decision threshold is fixed at **70 mg/dL** (the clinical hypoglycemia
cutoff) for both baselines.

### Naive Last Value
**Rule:** If the most recent glucose reading (last step in the lookback window)
is below 70 mg/dL → predict event.

This rule fires only when the patient is *already* in hypoglycemia.
It is highly precise (when it fires, it is usually right) but misses most
events because it cannot look ahead at all.

### Naive Mean
**Rule:** If the average glucose over the entire 2-hour lookback window
is below 70 mg/dL → predict event.

The mean needs to be very low to fall under 70, so this rule fires very
rarely and misses almost everything.

---

## Trained Models

### Model 1 — LSTM Event Classifier ("event in 60 min")

**Architecture:** LSTM (2 layers, 256 hidden units, baseline hyperparameters)

**What it predicts:**
> "Will glucose fall below 70 mg/dL *at any point* in the next 60 minutes?"

The label for each input window is positive (1) if the minimum of the
next 12 steps (60 min) is below 70 mg/dL. The model is trained with
`BCEWithLogitsLoss` and a positive class weight to handle the imbalance
caused by the rarity of hypo events.

**Key difference from the naive baselines:** the model can fire an alarm
even when glucose is currently still in the normal range, if it detects
a downward trend. This gives it higher recall (it catches more future events)
at the cost of more false alarms.

---

### Model 2 — PatchTST Event Classifier ("event in 60 min")

**Architecture:** PatchTST (patch-based transformer, baseline hyperparameters)

**What it predicts:**
> Same question as Model 1, same label definition.

This is the transformer-based counterpart of Model 1. It processes the
lookback window as a sequence of non-overlapping patches, which may or may
not be better suited for detecting the long-range trends that precede a
hypoglycemia event.

---

### Model 3 — LSTM Point-Event Classifier ("event at exactly 60 min")

**Architecture:** LSTM (identical to Model 1)

**What it predicts:**
> "Will glucose be below 70 mg/dL *specifically around the 60-minute mark*
> (within a ±15-minute tolerance window)?"

The label is positive if the minimum of the 7-step window
**[45 min, 75 min]** ahead is below 70 mg/dL (i.e. steps 9–15 from now).
This is a harder, more specific question than Model 1: the event must
happen *near* the target horizon, not just somewhere in the next hour.

The ±15-minute tolerance (D = ±3 steps) is analogous to the bounded-lag
loss in Objective 1 — it gives the model some timing slack around the
exact 60-minute target.

---

### Model 4 — PatchTST Point-Event Classifier ("event at exactly 60 min")

**Architecture:** PatchTST (identical to Model 2)

**What it predicts:**
> Same question as Model 3, same label definition.

---

## Lead-Time Analysis

For the "event in 60 min" classifiers (Models 1 and 2), we measured how
many minutes into the prediction window the *first* true hypo step actually
occurs, for every correct (True Positive) prediction.

| Model | True Positives | Mean time to event | Median time to event |
|---|---|---|---|
| LSTM Event | 2 610 | **12.4 min** | 5 min |
| PatchTST Event | 2 243 | **18.3 min** | 15 min |

**Interpretation:** For the LSTM, roughly half (49.5 %) of all True Positive
predictions have the event already occurring at the very first step of the
60-minute window (i.e. glucose was already below 70 when the alarm fired).
The model acts largely as a reactive detector of *ongoing* hypos rather than
a predictor of *future* ones.  PatchTST is somewhat better distributed,
with a median warning time of 15 minutes.

---

## D-Tolerance Sweep (Models 3 and 4)

For the point-event classifiers we swept the evaluation tolerance D from
±0 steps (exact 60-minute match only) to ±6 steps (±30 min), to see how
sensitive performance is to the size of the acceptance window.

**LSTM Point-Event:**

| D | Window | Precision | Recall | F1 |
|---|---|---|---|---|
| ±0 | 1 step — exact 60 min | 0.052 | 0.553 | 0.090 |
| ±1 | 3 steps — ±5 min | 0.064 | 0.536 | 0.109 |
| ±2 | 5 steps — ±10 min | 0.077 | 0.530 | 0.127 |
| **±3** | **7 steps — ±15 min (training)** | **0.088** | **0.525** | **0.143** |
| ±4 | 9 steps — ±20 min | 0.100 | 0.525 | 0.159 |
| ±6 | 13 steps — ±30 min | 0.124 | 0.526 | 0.190 |

Recall stays nearly flat across all D values (~0.52–0.55) while precision
rises only slowly. Widening the acceptance window helps slightly, but the
fundamental problem — too many false alarms — does not go away.

**PatchTST Point-Event:** Same pattern; precision 0.020–0.050 across all D.

---

## Results Summary

### Task A — "Will there be a hypo event anywhere in the next 60 min?"

| Metric    | Naive Last Value | Naive Mean | LSTM Event | PatchTST Event |
|-----------|-----------------|------------|------------|----------------|
| Precision | 0.857           | 0.396      | 0.237      | 0.054          |
| Recall    | 0.326           | 0.041      | **0.693**  | 0.615          |
| F1        | 0.473           | 0.074      | 0.334      | 0.095          |
| F2        | 0.373           | 0.050      | **0.463**  | 0.179          |

### Task B — "Will there be a hypo event at exactly 60 min (±15 min)?"

| Metric    | Naive Last Value | Naive Mean | LSTM Point-Event | PatchTST Point-Event |
|-----------|-----------------|------------|------------------|----------------------|
| Precision | 0.216           | 0.115      | 0.088            | 0.036                |
| Recall    | 0.113           | 0.016      | **0.525**        | 0.537                |
| F1        | 0.149           | 0.029      | **0.143**        | 0.064                |
| F2        | 0.125           | 0.020      | **0.237**        | 0.124                |

---

## Key Takeaways

**1. LSTM Event vs. Naive Last Value (Task A)**
The naive last-value rule achieves a higher F1 (0.473) than the LSTM (0.334),
but the comparison is not entirely fair: the two models operate at very
different precision/recall trade-offs. The naive rule fires only when glucose
is *already* below 70 (very precise, low recall), while the LSTM fires
earlier and more often (high recall 0.693, but many false alarms).
For a clinical alarm system, high recall is usually more important than high
precision — a missed hypo is more dangerous than a false alarm. On that
metric, the LSTM clearly wins (recall 0.693 vs. 0.326).

**2. PatchTST Event falls below the naive mean baseline (Task A)**
PatchTST achieves F1 = 0.095, which is lower than the simple mean rule
(F1 = 0.074 — close to, but slightly below). This indicates the PatchTST
event classifier has not learned anything reliably beyond what a simple
average already tells you.

**3. Point-event models (Task B) are harder than they appear**
Both LSTM and PatchTST achieve very low precision on Task B. The LSTM
barely exceeds the naive last-value baseline on F1 (0.143 vs. 0.149),
and PatchTST is clearly below it (0.064 vs. 0.149). Predicting the exact
60-minute mark is a significantly harder problem than predicting whether
*any* event will occur in the next hour.

**4. Models are reactive, not truly predictive**
The lead-time analysis and the naive baseline comparison both point to the
same conclusion: the models largely detect hypos that are *already happening*
or *about to happen in the next few minutes*, rather than warning the patient
well in advance.
