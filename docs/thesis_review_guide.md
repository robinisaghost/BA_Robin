# Thesis Review, Improvement & Verification Guide

> **How to use this:** Paste this whole file as the instruction set at the start of a
> Claude session, then point it at one chapter at a time. It encodes the project
> context, the writing standards agreed with the author, and a chapter-by-chapter
> checklist. Work incrementally: read the current state first, propose changes and
> explain the *reasoning* before applying large structural edits, and recompile to
> verify after each round.

---

## 1. Project context

- **Title:** *Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and
  Hypoglycemia Event Prediction* — BSc thesis, University of Bern (Pattern
  Recognition Group), supervisor PD Dr. Kaspar Riesen. Working language German;
  thesis text in English.
- **Core phenomenon:** Deep models trained with **MSE** on the **strongly
  autocorrelated** CGM signal minimise pointwise error by *reproducing recent
  history*, so predictions are displaced forward in time by ~50–55 min. This
  makes **hypoglycaemia** (glucose < 70 mg/dL) get detected too late or missed.
- **Controlled ablation — exactly one variable changes per objective:**
  - **Obj 1:** offset-aware losses — Bounded-Lag (fixed local tolerance) and
    **Soft-DTW** (elastic time-warping; the author's supervisor finds this the
    most interesting — give it the most depth).
  - **Obj 2:** multi-step supervision (predict all H=12 steps).
  - **Obj 3:** event-centric evaluation — tolerance-window (τ) sweep, direct
    binary classifiers, and a lead-time analysis.
- **Architectures:** LSTM and PatchTST (PatchTST uses **RevIN**). **Data:**
  T1DATA, **36 patients**, **trained per patient**. **Diagnostic metric:**
  RMSE\* (lag-adjusted RMSE). **Primary detection metric:** F2.
- **Files:** `thesis/{introduction,background,method,experiments,conclusions,abstract,appendix}.tex`,
  `thesis/lib.bib`, figures in `thesis/img/`, loss-figure script `generate_plots.py`.
- **Reference docs in `docs/`:** the PRG writing guide
  *"From Research to Thesis"* (the authority on style/structure), the
  `glucose_prediction_proposal.pdf` (defines the research questions), and
  `BA_FabianHuni.pdf` (prior PRG thesis on the same dataset).

## 2. How to compile & verify (Windows / MiKTeX)

Run from the `thesis/` directory. Full build = pdflatex → bibtex → pdflatex → pdflatex.

```powershell
$env:PATH = "C:\Users\Robin\AppData\Local\Programs\MiKTeX\miktex\bin\x64\;" + $env:PATH
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

After building, check `main.log` for problems:
- `Citation .* undefined` / `Reference .* undefined` → broken \cite or \ref.
- `File .* not found` → missing figure.
- `Rerun to get` → run pdflatex once more.

A clean build is currently **57 pages**. Never report "done" without a clean compile.

## 3. Writing standards (treat as non-negotiable)

**Academic phrasing**
- No vague intensifiers: avoid *extremely, very, hugely, dramatically*. Use precise
  or comparative terms (*marginal, considerable, near-zero, markedly lower*).
- `significant` / `significantly` are reserved for **statistical** significance and
  must be backed by a test. There is **no statistical test** in this thesis (a
  Wilcoxon test was deliberately removed). Use *considerable, pronounced, notable*.
- `substantially outperform` and similar: only with explicit, defensible comparison.

**Evidence & citations (credibility)**
- Anything taken from another source must be cited. A claim with a number or a
  mechanism needs a source or the author's own computation.
- **Clinical-severity / cost-asymmetry** claims → `battelino2019`.
- **MSE rewards copying history on autocorrelated signals** → `wolff2025cgpm`.
- **Standard ML losses (MSE, BCE)** → `goodfellow2016deeplearning`.
- **Direct hypoglycaemia classifiers** → `shao2024hypoglycemia, cinar2025hypoclassification`.
- **Meals drive glucose excursions** → `sun2026futureaware`.
- In **Conclusions**, reference the internal section where a source was already
  cited rather than re-citing; only a *newly introduced* external claim needs its
  own citation there.
- Do **not** invent sources or numbers. If a value isn't computed or cited, either
  compute it (e.g., the lag-1 autocorrelation = 0.985 was computed from
  `data/raw/all_cgm.csv`) or remove the specific figure.

**Clinical caution (this loses points if overdone)**
- Avoid unsupported clinical-utility claims: *clinically meaningful, clinically
  useful, clinical deployment, early-warning system, actionable for a patient*.
- State the factual consequence instead (e.g., "the alarm leaves no interval ahead
  of the event"), not an efficacy claim. Limitations may note caveats neutrally.

**Figures & tables (must earn their space)**
- Every figure/table is referenced **and actively described** in the running text —
  walk the reader through what it shows, don't just write "see Fig. X".
- Text must **not restate the numbers** that are already in a table; it interprets
  them and explains relationships. The reader can read the table.
- A caption must be self-explanatory (what is shown *and* why). Describe line
  colours/roles where relevant.

**Structure (per the PRG guide)**
- Max **three** subsections per section. If more are needed, group them (e.g., the
  two offset-aware losses live under one "Offset-Aware Losses" subsection as
  `\subsubsection`s).
- A subsection level needs ≥ 2 entries to exist.
- Theory/motivation lives in **Ch 2–3**; application/results in **Ch 4–5**. Avoid
  overlap: one home per concept (e.g., the weighted-BCE *formula* is in Ch 2.2.3;
  Ch 3 only states the applied weight and evaluation).

**The describe → interpret → discuss split**
- **Ch 4 describes and provokes thought** (observations, "this suggests…",
  "raises the question…"). It must **not** draw the final conclusions and must not
  assert mechanisms as settled (e.g., do *not* explain the inversion via RevIN in
  Ch 4).
- **Ch 5 draws the conclusions** and may state mechanisms (RevIN explanation lives
  here), grounded in the Ch 4 observations.

## 4. The red thread (verify it is intact end-to-end)

The three research questions are introduced in **Ch 1**, operationalised in
**Ch 3.3** (each objective explicitly tied to its RQ), addressed in **Ch 4**, and
**answered** in **Ch 5**. Check this chain is unbroken and consistent:

- **RQ1 (offset-aware losses):** do they reduce the delay / improve detection?
  → Answer: largely **no**. Even Soft-DTW (the most flexible) raises RMSE\*;
  realigning in time costs shape. The delay is **structural**, not loss-fixable.
- **RQ2 (multi-step):** → **partial** — best trajectories but the lag is unchanged.
  First evidence that **trajectory quality and temporal alignment are dissociable**
  (this should be confirmed against `fox2018multioutput`, cited in Ch 1).
- **RQ3 (direct classifiers):** → **yes** for detection (F2 ≈ 0.46 vs < 0.09), but
  two qualifications: the **architecture inversion** (RevIN removes the absolute
  scale a threshold classifier needs) and the **lead-time** finding (≈ half the
  LSTM alarms are reactive). **F2 ≠ anticipation.**
- **Synthesis:** forecasting and detection are **separate tasks**; the delay can't
  be optimised away at the loss level.

## 5. Chapter-by-chapter checklist

**Ch 1 — Introduction**
- [ ] T1D context → CGM/hypoglycaemia → time-shift problem → consequence → related
  work → setup → RQs → contributions → thesis outline. Each paragraph one message.
- [ ] Cost-asymmetry (missed > false alarm) is cited (`battelino2019`).
- [ ] RMSE\* described conceptually on first mention, the *symbol* deferred.
- [ ] Related-work demarcation is explicit: Wolff/Sun **diagnose** the problem;
  this thesis **evaluates mitigations** (don't claim the problem as novel).
- [ ] Numbers consistent: 36 patients, 10 configurations, ~50–55 min delay.

**Ch 2 — Background**
- [ ] Architectures (LSTM, PatchTST) with figures actively walked through; tuned
  hyperparameter values deferred to Ch 3 Table (not duplicated here).
- [ ] **RevIN framing:** it counters *within-series non-stationarity* — NOT
  "heterogeneous patient profiles" (training is per-patient).
- [ ] Losses grouped by role: 2.2.1 MSE, 2.2.2 Offset-Aware (Bounded-Lag +
  Soft-DTW), 2.2.3 Weighted BCE. **Soft-DTW deepest**; Bounded-Lag concise/framed
  as conservative; MSE gives the *mathematical* basis of the shift.
- [ ] Every loss formula and Fig 2.3/2.4 actively integrated.
- [ ] Metrics = exactly 3 subsections; defines every Ch 4 table column incl. the
  "Lag" column; F1 vs F2 made explicit.

**Ch 3 — Method**
- [ ] Section 3.2 = "Dataset and Experimental Setup" (covers splits + hyperparams).
- [ ] Lag-1 autocorrelation stated as the computed cohort value (0.985, SD 0.029).
- [ ] Class-imbalance paragraph: clear *why* imbalance is a problem + the two
  countermeasures (F2, weighted BCE).
- [ ] Direct classifier fully defined (sigmoid head, weighted BCE → Ch 2.2.3,
  per-window evaluation at p̂ > 0.5).
- [ ] Each objective states its research question (consistent format).

**Ch 4 — Empirical Evaluation**
- [ ] Per result section: text → figure → text → table, with active figure/table
  description and **no number-dumping**.
- [ ] **Describe & provoke, don't conclude.** Use "suggests / is consistent with /
  raises the question". No RevIN mechanism asserted here.
- [ ] Soft-DTW (4.2.2) gets more space/depth than Bounded-Lag (4.2.1).
- [ ] Bold = best value per **column** in every results table.
- [ ] fox2018multioutput confirmation present in 4.3.

**Ch 5 — Conclusions**
- [ ] Each RQ restated and answered, grounded in Ch 4, each paragraph one message.
- [ ] tau-sweep diagnostic present (LSTM structural vs PatchTST timing failure).
- [ ] Clinical claims softened. Tuning caveat framed so the *ablation conclusions
  stay robust* (all variants share the fixed config).
- [ ] Future work tied to limitations: richer inputs (incl. time-of-day),
  RevIN/scale (Soft-DTW + RevIN ablation), data-centric (informed sampling +
  imbalance). **No hyperparameter-tuning direction.**

**Abstract**
- [ ] ~250–300 words, one page, **no citations**, self-contained.
- [ ] Tells the red thread (problem → 3 findings → separate-tasks conclusion);
  headline numbers only; no clinical-use claim.

## 6. Decisions already made — do NOT silently undo

- 0.98 autocorrelation was hallucinated → replaced by the computed 0.985 in Ch 3.
- The Wilcoxon/Holm significance test and its appendix were **removed** on purpose.
- T1DATA reference `garciatirado2023t1d` is **correct**; cohort is **36** patients.
- RMSE\* is introduced in Ch 2 and used directly thereafter (don't re-introduce it).
- Per-patient training is fundamental — keep RevIN framed accordingly.

## 7. Working method

- Read the current file/section before editing; don't rewrite from memory.
- For structural changes, explain the reasoning and get agreement first.
- Make minimal, targeted edits; keep surrounding style/voice.
- After each round, **recompile** and check the log; report honestly (page count,
  any warnings). If something is unverified, say so.
