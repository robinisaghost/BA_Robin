# Speaker Notes — PRG Seminar Presentation
# Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction
# Robin van den Hoek | University of Bern | 30 min

---

## Slide 1 — Title

Good morning / afternoon everyone. My name is Robin van den Hoek, and this is my bachelor thesis
in the Pattern Recognition Group. My thesis is titled "Mitigating Time-Shift Errors in
CGM-based Glucose Forecasting and Hypoglycemia Event Prediction," supervised by PD Dr. Kaspar
Riesen. I'll be presenting the full project today in about 30 minutes.

---

## Slide 2 — Motivation: The Diabetes Problem

Let me start with why this problem matters clinically.

Diabetes is one of the most widespread chronic diseases worldwide. According to the IDF Diabetes
Atlas [1], 537 million adults were living with diabetes in 2021, and that number is projected to
reach 783 million by 2045. Of these, people with Type 1 diabetes are of particular relevance here:
in Type 1, the pancreas produces no insulin at all, meaning the patient must manage blood glucose
manually or via an automated system throughout the day.

The danger I'm focusing on is hypoglycemia — blood glucose dropping below 70 milligrams per
deciliter. A hypoglycemic episode can cause confusion, loss of consciousness, and in severe cases
death. The key challenge is that by the time a patient feels symptoms, it may already be too late
to act.

This is the clinical motivation for predictive algorithms: if we can forecast glucose 60 minutes
ahead, we can trigger a preventive action — adjusting an insulin pump, alerting the patient —
before the episode occurs.

---

## Slide 3 — Motivation: CGM Technology

The data source I'm working with is Continuous Glucose Monitoring — CGM. A CGM sensor, such as
the Dexcom G6 [2], is worn on the body and measures blood glucose every 5 minutes continuously.
This produces a real-time time series that can be fed into a machine learning model.

In a closed-loop insulin delivery system — sometimes called an artificial pancreas — the CGM
provides the input signal, and a control algorithm determines when and how much insulin to deliver.
The bottleneck in such systems is the prediction component: the algorithm needs to know not just
the current glucose value, but what glucose will be in the near future. This is where my work
fits in.

---

## Slide 4 — Motivation: Why 60-min Prediction?

Why specifically 60 minutes? The pharmacokinetics of insulin are the answer. Insulin takes
approximately 60 to 90 minutes to act once injected. So if a model detects a downward glucose
trend *now*, a preventive insulin reduction or correction must be initiated now — not when the
patient is already symptomatic.

The setup I use throughout this thesis is: given a 2-hour lookback window of glucose measurements
(24 steps at 5-minute intervals), predict the glucose value at exactly 60 minutes in the future
(step h=11 out of a 12-step horizon). The dataset is T1DATA [3] (Garcia-Tirado et al.): 36 patients
with Type 1 diabetes, measured over approximately 30 days at 5-minute intervals. I train one model per
patient — 36 independent models for each architecture.

---

## Slide 5 — Scope

Let me be explicit about scope before going further.

I use a univariate approach: only glucose values as input — no insulin, no carbohydrate intake,
no activity data. This is an important constraint. Many real-world systems use additional signals,
but the PRG Proposal [7] specifically focuses on the glucose signal alone to isolate the forecasting
problem.

I compare two architectures throughout: an LSTM, as the standard recurrent baseline for time
series, and PatchTST, a state-of-the-art Transformer model for time series that was introduced by
Nie et al. [4] in 2023.

What I am NOT covering: multi-modal input, closed-loop control, population-level generalization,
or patient transfer. This is a controlled ablation study on a single dataset.

---

## Slide 6 — The Time-Shift Problem

Now for the central observation that motivates the entire thesis.

When you train an LSTM or PatchTST with standard MSE loss on the 60-minute prediction task, you
get predictions that look like this [refer to patient plot on slide]. The predicted curve — shown
in orange — follows the shape of the true glucose curve beautifully. But it is systematically
delayed. It looks like someone copy-pasted the input signal and shifted it forward by
approximately an hour.

Quantitatively: the LSTM has a mean temporal shift of 10.2 steps, or 51 minutes. PatchTST: 11.0
steps, or 55 minutes. This is almost the entire prediction horizon.

To measure this, I use a diagnostic metric I call lag_rmse: the RMSE after applying the optimal
constant temporal shift to the prediction. It tells you: if we could correct the timing, what
RMSE would remain? A low lag_rmse means the prediction is accurate in shape; a high shift penalty
(RMSE minus lag_rmse) means most of the error comes from timing, not shape.

For the LSTM baseline: RMSE = 36.09, lag_rmse = 21.65 — so about 14.44 mg/dL of error is purely
timing. For PatchTST: RMSE = 39.27, lag_rmse = 14.21 — a whopping 25 mg/dL of timing error.

This is the central pathology: models learn to reproduce the glucose trend rather than anticipate
future changes. This is what the three objectives of this thesis attempt to address.

---

## Slide 7 — Baseline Metrics

Before introducing the objectives, here are the full baseline results.

The two models perform differently across metrics. LSTM is better at pointwise accuracy (lower
RMSE, 36.09 vs 39.27), while PatchTST has a much lower lag_rmse (14.21 vs 21.65), meaning
PatchTST's shape is more accurate once you remove the timing component. PatchTST also performs
better at hypoglycemia event detection from the baseline forecasts: Hypo-F1 of 0.061 versus
0.005 for LSTM.

Both values are very low in absolute terms — neither model is clinically useful for hypoglycemia
detection based on forecast threshold crossings alone. This is the problem Objective 3 will
address directly.

I'll reference these baseline numbers throughout the talk as the comparison point.

---

## Slide 8 — Model Architecture: LSTM

The LSTM architecture is straightforward: a stacked LSTM with two layers, hidden size 256,
followed by a linear head that maps the final hidden state to a single scalar — the predicted
glucose at 60 minutes.

Hyperparameters were tuned using Optuna [7] with 50 trials on a single tuning patient (patient
85202), then applied identically to all 36 patients. This is critical: the same hyperparameters
are used for all three objectives. The only thing that changes between objectives is the loss
function or output head — never the architecture or hyperparameter search space. This ensures
controlled ablation experiments.

Training uses MSE loss, AdamW optimizer, and early stopping with patience 10.

---

## Slide 9 — Model Architecture: PatchTST

PatchTST [4], published at ICLR 2023, introduces a Transformer-based model for time series that
works on patches rather than individual time steps.

The input window is divided into overlapping patches — here: patch length 12 steps, stride 6,
giving 3 patches from a 24-step lookback window. Each patch is embedded and fed into a standard
Transformer encoder. The output passes through a linear head to produce the 60-minute prediction.

A key component is Reversible Instance Normalization, or RevIN: the input is normalized
per-sample before the Transformer (subtracting the sample mean, dividing by std), and
denormalized after the prediction. This helps the model generalize across patients with different
glucose baseline levels.

Hyperparameters: d_model=64, 4 attention heads, 3 Transformer layers, feed-forward dimension 256,
dropout 0.025, learning rate 1.21e-4. Same Optuna procedure as LSTM.

---

## Slide 10 — Project Objectives Overview

The three objectives each target a different aspect of the time-shift problem.

Objective 1 changes the *loss function*: instead of penalizing each predicted step against its
exact true position, we introduce a tolerance for small temporal offsets during training.

Objective 2 changes the *training target*: instead of predicting only the 60-minute step, we
supervise the full 12-step horizon simultaneously.

Objective 3 reframes the *evaluation task*: rather than deriving event detection from a
continuous forecast, we train a dedicated binary classifier for hypoglycemia events.

All three objectives use the exact same architectures and the same Optuna-tuned hyperparameters as
the baseline. One variable changes at a time.

---

## Slide 11 — Objective 1: Motivation and Approach

Standard MSE loss treats every step independently. If the prediction is off by 10 mg/dL at
exactly the right time, the penalty is 100. But if the prediction is off by 10 mg/dL at a time
step that is 15 minutes late — which is clinically nearly equivalent — the penalty is the same
100. The loss has no awareness of temporal structure.

Objective 1 tests two approaches to fix this.

First, Bounded-Lag Loss (1a): for each prediction step, find the best matching true value within
a window of D=3 steps (±15 minutes) and compute MSE only against that best match. This gives the
model a tolerance of ±15 minutes in training.

Second, Soft-DTW Loss (1b) [5]: Differentiable Dynamic Time Warping applies elastic alignment
along the entire sequence, smoothed by a parameter gamma=1.0. This is a well-known approach in
time series learning that allows the model to match shapes without being penalized for small
temporal offsets.

---

## Slide 12 — Objective 1: Results

Here are the results for both loss variants on both architectures.

[Table on slide]

For LSTM: Bounded-Lag adds only 0.05 mg/dL to RMSE — essentially no cost. Soft-DTW increases
RMSE by 1.14 mg/dL. Both slightly improve Hypo-F1 from 0.005 to 0.014. Neither improves
lag_rmse relative to baseline — in fact, both worsen it (25.25 and 28.67 vs 21.65 baseline).

For PatchTST: Bounded-Lag adds only 0.03 mg/dL RMSE. Soft-DTW adds 3.96 mg/dL — a significant
regression. Again, both worsen lag_rmse relative to baseline (16.70 and 24.01 vs 14.21 baseline).

---

## Slide 13 — Objective 1: Interpretation

Why don't the offset-aware losses help?

The core problem is scale. The D=3 step tolerance window in Bounded-Lag covers ±15 minutes. But
the actual shift is 50–55 minutes. We are tolerating 15 minutes of offset in training, but the
model is already delayed by 55 minutes. The tolerance window is too small to make a structural
difference.

For Soft-DTW: elastic alignment helps during training but does not give the model structural
information about when changes will occur. The model can align shapes at training time, but
at test time it still has no signal about future glucose dynamics — so it defaults to the
same lag-1 autocorrelation strategy.

The higher lag_rmse for bounded-lag and DTW models (compared to baseline) suggests that relaxing
the penalty for timing actually allows the model to become more variable in its timing errors —
making it harder to compensate with a single constant shift.

The key lesson: loss relaxation without additional structural supervision is insufficient.

---

## Slide 14 — Objective 2: Motivation and Approach

Objective 2 takes a different approach: change what the model is supervised on.

Instead of predicting only the 60-minute endpoint (h=11), we predict all 12 steps simultaneously.
This is the DIRMO strategy — Direct Multi-Output. The loss function averages MSE over all 12
output positions (steps 0 through 11, corresponding to 5 through 60 minutes). Evaluation is still
performed at h=11 for direct comparability with the baseline.

The hypothesis is: if the model must correctly predict intermediate steps — including short
horizons of 5 or 10 minutes — it cannot shift its entire prediction later, because those early
steps would be penalized. The multi-step constraint forces the model to stay phase-aligned
throughout the forecast horizon.

The architecture change is minimal: the output head changes from (B, 1) to (B, 12). No other
change. Same hyperparameters.

---

## Slide 15 — Objective 2: Results

[Table on slide]

Multi-step forecasting is the best approach across all methods tested.

For LSTM: RMSE drops from 36.09 to 35.59 — a 0.50 mg/dL improvement. lag_rmse drops from
21.65 to 20.74. Hypo-F1 improves from 0.005 to 0.010.

For PatchTST: RMSE drops from 39.27 to 39.05 — a 0.22 mg/dL improvement. lag_rmse: 14.21
baseline vs 14.53 multi-step, essentially unchanged. Hypo-F1 improves from 0.061 to 0.072.

The shift penalty — RMSE minus lag_rmse — for LSTM remains at 14.85 (vs 14.44 baseline), so the
proportion of error attributable to timing has not dramatically changed.

---

## Slide 16 — Objective 2: Interpretation

Multi-step is the most consistent improvement across both models, but the gains are modest.

For LSTM, the lag_rmse reduction (from 21.65 to 20.74) is meaningful — it's the only method
that actually reduces lag_rmse. For PatchTST, the improvement is negligible.

Why doesn't multi-step eliminate the shift entirely? Because the model still fundamentally lacks
access to information about future glucose dynamics. The lookback window contains 2 hours of
past glucose. The upcoming 60 minutes depend heavily on factors not in the signal: meals,
activity, insulin — none of which appear in our univariate input. The intermediate step
supervision prevents the model from being completely lazy, but cannot compensate for the
fundamental information deficit.

Multi-step is strictly better than Objective 1 approaches and should be the default when training
these models. But the time-shift problem is not solved.

---

## Slide 17 — Objective 3: Motivation

Given that forecasting-based methods all show limited success at reducing the temporal shift, I
step back and ask: what does the clinician actually need?

A clinician managing a Type 1 diabetes patient doesn't necessarily need a perfect glucose curve.
They need to know: will this patient experience hypoglycemia in the next hour? That is a binary
question: yes or no.

Objectives 1 and 2 optimize MSE — a metric that penalizes pointwise deviations regardless of
clinical significance. Event detection accuracy is only measured indirectly, via threshold
crossings in the predicted glucose series. MSE-trained models are not calibrated for event
detection.

Objective 3 has two parts. First, I quantify how bad the baseline event detection really is using
a tolerance sweep. Then, I train a dedicated binary classifier that directly answers the yes-or-no
question.

---

## Slide 18 — Objective 3a: The τ-Sweep

For clinical hypoglycemia alerting, there is a natural tolerance: an alert 10 minutes early is
still useful. An alert 20 minutes early is still useful. The clinical system just needs the
alert before the event, not at the exact millisecond.

The τ-sweep measures F1, Precision, and Recall for hypoglycemia event detection at varying time
tolerances τ ∈ {0, 1, …, 12 steps}. A predicted threshold crossing is counted as a True Positive
if it falls within ±τ steps of the true crossing.

[Point to figure]

The left panel shows F1 vs. τ. LSTM (blue) stays very low — even at τ=12 (±60 min tolerance),
F1 only reaches 0.113. PatchTST (orange) improves dramatically from near 0 at τ=0 to F1=0.551
at τ=12.

What does this tell us? PatchTST is detecting *something* near the right events — but with large
temporal offsets matching its 55-minute mean lag. When the tolerance window matches the model's
own lag, the apparent F1 rises sharply. LSTM rarely detects events at all because its predicted
curve almost never crosses the 70 mg/dL threshold.

At the clinically relevant τ=3 (±15 min): LSTM F1=0.005, PatchTST F1=0.061. Both fail.

---

## Slide 19 — Objective 3b: Direct Binary Classifier

The second part of Objective 3 is to build a dedicated binary classifier.

Instead of predicting a full glucose curve and then checking if any predicted value crosses 70
mg/dL, I replace the output head with a single sigmoid neuron. The task is:
"given the 2-hour lookback window, will hypoglycemia occur at any point in the next 60 minutes?"

The label is 1 if any value in y_{t+0} through y_{t+11} is below 70 mg/dL. Loss function:
Binary Cross-Entropy with logits (BCEWithLogitsLoss). The LSTM and PatchTST backbones are
identical to the baseline — only the output head changes. Same Optuna hyperparameters.

---

## Slide 20 — Objective 3b: Results

The results are striking.

[Table on slide]

The LSTM binary classifier achieves F1=0.334 — a 62-fold improvement over the forecast-derived
baseline (F1=0.005). Recall=0.693 means nearly 70% of hypoglycemia events are correctly predicted.
Precision=0.237 — roughly 1 in 4 positive alerts is a true event.

The PatchTST binary classifier achieves F1=0.095 — only a 1.6-fold improvement. Despite recall
of 61.5%, precision is only 5.4%.

This asymmetry between the two architectures is worth examining carefully — which is what the
next slide does.

---

## Slide 21 — Why Does PatchTST Have F1=0.095 Despite Recall=61.5%?

Let me explain this mathematically first. F1 = 2 × (P × R) / (P + R).
With P=0.054 and R=0.615: F1 = 2 × 0.054 × 0.615 / (0.054 + 0.615) = 0.099.
So the value 0.095 is mathematically consistent — it follows directly from the low precision.

The problem is precision: for every 18 positive predictions PatchTST makes, only 1 is a true
hypoglycemia event.

Per-patient evidence confirms the scale of this problem. For Patient 85101: 29 true positives
versus 1284 false positives — a ratio of 1:44. The model is firing on almost every window it
sees.

Three root causes:

First, class imbalance. Hypoglycemia is rare. Even in a Type 1 diabetes cohort, most CGM windows
do not contain a hypoglycemic event. A model that predicts "hypoglycemia" on every window would
achieve high recall but near-zero precision. This prior dominates when the classifier is not
well-calibrated.

Second, architectural mismatch. PatchTST's attention mechanism operates over patches and detects
local patterns. Any descending glucose trend resembles a potential hypoglycemia event to the
Transformer, even if the descent is mild and glucose stays safely above 70. The LSTM's recurrent
state integrates information over time and is more conservative: it has an implicit "prior" about
how long trends persist before a threshold crossing occurs.

Third, no threshold tuning. Both classifiers use the default sigmoid threshold of 0.5. Under
class imbalance, the optimal threshold is much lower — a precision-recall curve analysis would
reveal the right operating point.

The comparison to LSTM proves this is architectural: same task, same labels, same threshold,
LSTM precision = 0.237 — 4.4× higher. This is not about luck in training; it's about how the
two architectures process sequential patterns.

---

## Slide 22 — Objective 3: Key Finding

To summarize Objective 3:

Separating forecasting from event detection is by far the most impactful approach tested in this
thesis. The LSTM binary classifier with F1=0.334 represents a genuinely meaningful step toward
clinical usefulness — Recall=0.693 means most hypoglycemia episodes would trigger an alert.

PatchTST's low F1 despite high recall is a cautionary result: raw recall alone is not sufficient
if precision is negligible. A clinical system that fires on every descending trend is not useful.

Class imbalance is a fundamental challenge that threshold tuning, focal loss, or class-weighted
training could address in future work.

The τ-sweep result strengthens the case: even with generous tolerance, MSE-trained forecasting
models barely improve. The forecasting objective and the event detection objective are simply
misaligned. Different models are needed for different tasks.

---

## Slide 23 — Summary: All Results

This table shows all methods and metrics side by side. Let me highlight the key comparisons.

For RMSE, multi-step is best: LSTM 35.59, PatchTST 39.05.

For lag_rmse, PatchTST baseline and multi-step are essentially tied at ~14.2–14.5, both better
than all other variants.

For Hypo-F1, the LSTM classifier dominates with 0.334 — 62× the nearest forecasting-based
approach. PatchTST classifier achieves only 0.095 due to the precision problem.

No single model dominates all metrics. Multi-step wins for forecasting; LSTM classifier wins for
hypoglycemia alerting.

---

## Slide 24 — Conclusions

Let me draw four conclusions from this work.

First: the time-shift problem (50–55 min mean lag) persists across all tested approaches. It
cannot be eliminated with the available univariate signal. Loss function relaxation (Obj 1) does
not help. Multi-step supervision (Obj 2) helps marginally for LSTM.

Second: multi-step forecasting (Obj 2) is the best strategy for reducing lag_rmse and improving
RMSE. It is also the simplest: just change the output head and average the loss over the horizon.

Third: MSE-based forecasting is fundamentally misaligned with event detection. Even with generous
time tolerances in the τ-sweep, F1 remains low for LSTM. Using the forecast to infer events is
not a viable strategy.

Fourth: direct binary classification (Obj 3b) achieves clinically meaningful results for LSTM
(F1=0.334, Recall=0.693). Two separate models are the recommendation: one for forecasting
(PatchTST or LSTM multi-step), one for hypoglycemia alerting (LSTM binary classifier).

---

## Slide 25 — Future Work

Several directions could improve on these results.

Threshold optimization: both classifiers use sigmoid threshold 0.5. A proper precision-recall
curve analysis could find the clinically appropriate operating point — especially for PatchTST
where a higher threshold would recover precision.

Class-weighted or focal loss during classifier training could counteract the class imbalance
directly at the loss level.

Multi-modal input: adding insulin records and carbohydrate intake would provide the missing
causal signal that drives glucose dynamics. This is the single most impactful architectural
change possible, but it was out of scope for this thesis.

Larger D in the Bounded-Lag loss (D=6 or D=8 steps instead of D=3) would cover a larger
fraction of the actual temporal offset. The current D=3 covers only ~30% of the 50–55 min shift.

Population-level model: training a single model on all patients and evaluating with leave-one-out
would test generalization, which per-patient models cannot assess.

---

## Slide 26 — Take-Home Messages

Three things I want you to remember:

One: ML glucose forecasting learns to copy the input signal with a delay — it gets the shape
right but shifts it by about one hour. This is a structural artifact of univariate MSE training
at long horizons.

Two: loss function changes alone cannot fix this. Only multi-step supervision provides any
improvement, and even that is marginal. The time shift is primarily an information problem, not
a loss design problem.

Three: forecasting and event detection are fundamentally different tasks that need different
models. A classifier trained directly for binary event detection outperforms the forecast-derived
approach by 62× in F1 for LSTM.

---

## Slide 27 — Discussion Questions

I'd like to leave a few minutes for discussion. Two questions I found particularly interesting
while working on this:

First: Could a model trained on the *rate of change* of glucose — dG/dt — instead of raw
glucose values avoid the time-shift artifact? The lag-1 autocorrelation is very high for raw
glucose but much lower for first differences. If the model learns on differences rather than
levels, it might not default to the "copy the past" strategy.

Second: How would you calibrate the tradeoff between false alarms and missed hypoglycemia events
in a clinical deployment? We used F2 to weight recall more heavily. But what F2 threshold would
you actually set — and who makes that decision, the developer or the regulatory body?

Thank you.

---

## Slide 28 — References

[1] International Diabetes Federation. IDF Diabetes Atlas, 10th edition. Brussels: IDF, 2021.
    Available at: diabetesatlas.org

[2] Dexcom Inc. Dexcom G6 Continuous Glucose Monitoring System. San Diego: Dexcom Inc., 2020.

[3] Garcia-Tirado J, Colmegna P, Villard O, et al. Assessment of Meal Anticipation for
    Improving Fully Automated Insulin Delivery in Adults With Type 1 Diabetes.
    Diabetes Care. 2023;46(9):1652-1658.

[4] Nie Y, Nguyen NH, Sinthong P, Kalagnanam J. A Time Series is Worth 64 Words:
    Long-term Forecasting with Transformers. In: Proceedings of the International Conference
    on Learning Representations (ICLR). 2023.

[5] Cuturi M, Blondel M. Soft-DTW: a Differentiable Loss Function for Time-Series.
    In: Proceedings of the 34th International Conference on Machine Learning (ICML).
    Sydney, Australia; 2017. pp. 894-903.

[6] Lim B, Zohren S. Time-series forecasting with deep learning: a survey.
    Philosophical Transactions of the Royal Society A. 2021;379(2194):20200209.

[7] Pattern Recognition Group, University of Bern. Glucose Prediction — Project Proposal.
    Internal unpublished manuscript. Supervisor: PD Dr. Kaspar Riesen; 2025.
