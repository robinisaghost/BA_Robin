# Ergebniszusammenfassung: Bachelor-Thesis

**Student:** Robin van den Hoek (22-127-641)  
**Supervisor:** PD Dr. Kaspar Riesen, Universität Bern (INF)  
**Besprechung:** 1. Mai 2026  
**Thema:** Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction

---

## Baseline

**Aufbau:** 36 Patienten, LSTM und PatchTST, Lookback=24 Schritte (2 h), Horizont h=11 (60 min), MSE-Verlustfunktion. Hyperparameter via Optuna auf Patient 85202 optimiert; alle nachfolgenden Objectives verwenden dieselben Hyperparameter (eine Variable ändern).

**Zentraler Befund:** Beide Modelle reproduzieren den Glukoseverlauf mit einer systematischen zeitlichen Verschiebung von ~50–55 Minuten. Die Vorhersage ist in Form korrekt, aber um fast den gesamten Prädiktionshorizont nach hinten verschoben.

**Diagnostische Metrik – lag_rmse:** Minimaler RMSE nach Anwendung einer optimalen konstanten Verschiebung k* ∈ [−12, 12] auf die Vorhersage. Gibt an, wie gross der Fehler wäre, wenn die zeitliche Verschiebung bereits korrigiert wäre.

| Modell | RMSE (mg/dL) | MAE (mg/dL) | Hypo-F1 | Ø Verschiebung |
|--------|-------------|------------|---------|----------------|
| LSTM MSE | 36.09 | 27.12 | 0.005 | 10.2 Schritte (51 min) |
| PatchTST MSE | 39.27 | 29.00 | 0.061 | 11.0 Schritte (55 min) |

**Fazit Baseline:** LSTM ist bei der Punktvorhersage genauer (RMSE), PatchTST zeigt marginal bessere Hypoglykämieerkennung aus der Prognose. Beide Modelle versagen bei der Früherkennung von Hypoglykämieereignissen (F1 < 0.07). Die Zeitverschiebung ist das zentrale Problem der Thesis.

---

## Objective 1 — Verschiebungstolerante Verlustfunktionen

**Ansatz:** Die Standard-MSE-Verlustfunktion bestraft jeden Zeitschritt einzeln, ohne Berücksichtigung zeitlicher Struktur. Zwei alternative Verlustfunktionen reduzieren die Strafe für kleine zeitliche Fehlausrichtung beim Training.

- **Bounded-Lag (1a):** `L = min_{Δ ∈ [−3, 3]} MSE(ŷ, y_shifted_Δ)` — bewertet die Vorhersage am besten passenden Offset innerhalb D=3 Schritte (±15 min).
- **Soft-DTW (1b):** Differenzierbares Dynamic Time Warping mit Glättungsparameter γ=1.0 — erlaubt elastische Zeitausrichtung entlang dem gesamten Sequenzpfad.

| Modell | Verlust | RMSE | lag_rmse | Shift-Strafe | Hypo-F1 |
|--------|---------|------|----------|--------------|---------|
| LSTM | MSE (Baseline) | 36.09 | — | — | 0.005 |
| LSTM | Bounded-Lag (D=3) | 36.14 | 25.25 | 10.89 | 0.014 |
| LSTM | Soft-DTW (γ=1) | 37.23 | 28.67 | 8.56 | 0.014 |
| PatchTST | MSE (Baseline) | 39.27 | — | — | 0.061 |
| PatchTST | Bounded-Lag (D=3) | **39.30** | **16.70** | 22.59 | 0.067 |
| PatchTST | Soft-DTW (γ=1) | 43.24 | 24.01 | 19.23 | 0.062 |

*Shift-Strafe = RMSE − lag_rmse*

**Wichtigste Erkenntnisse:**

- **Bounded-Lag:** Nahezu kein RMSE-Verlust (Δ<0.05 mg/dL). PatchTST erreicht den niedrigsten lag_rmse aller Objective-1-Varianten (16.70). Leichte F1-Verbesserung für LSTM (0.005→0.014).
- **Soft-DTW:** RMSE-Regression, besonders bei PatchTST (+3.96 mg/dL). lag_rmse höher als Bounded-Lag. Die elastische Ausrichtung begünstigt Form über Amplitude – ein bekanntes Problem bei DTW-Verlusten auf Regressionsdaten.
- **Grundproblem beider Ansätze:** Das Fenster D=3 Schritte (15 min) deckt nur ~30% der tatsächlichen Verschiebung (50–55 min) ab. Die Verlustfunktion reduziert die Trainingsstrafe für kleine Fehlausrichtungen, gibt dem Modell aber keine strukturelle Information, um früher vorherzusagen.

---

## Objective 2 — Multi-Step Forecasting (H=12)

**Ansatz:** Statt eines einzelnen Werts am Horizont h=11 werden alle 12 Schritte (0–60 min) gleichzeitig vorhergesagt (DIRMO-Strategie). Die Verlustfunktion mittelt den MSE über alle 12 Ausgabepositionen. Evaluation weiterhin bei h=11 für direkte Vergleichbarkeit mit Baseline.

**Hypothese:** Das Vorhersagen der Zwischenschritte zwingt das Modell, über den gesamten Horizont in Phase zu bleiben. Zeitliche Drift wird bereits bei frühen Schritten bestraft.

| Modell | RMSE | MAE | lag_rmse | Shift-Strafe | Hypo-F1 |
|--------|------|-----|----------|--------------|---------|
| LSTM Multi-step | 35.59 | 26.80 | 20.74 | 14.85 | 0.010 |
| PatchTST Multi-step | 39.05 | 28.84 | 14.53 | 24.52 | 0.072 |

**Wichtigste Erkenntnisse:**

- **Niedrigstes lag_rmse aller getesteten Methoden** (beide Architekturen): PatchTST 14.53, LSTM 20.74.
- Leichte RMSE-Verbesserung gegenüber Baseline (LSTM: −0.50, PatchTST: −0.22 mg/dL).
- Die Shift-Strafe bleibt jedoch mit ~40–63% des RMSE erheblich → die strukturelle Zeitverschiebung wird reduziert, aber nicht beseitigt.
- Multi-step MSE ist der beste Ansatz für reine Prognosegenauigkeit unter allen getesteten Methoden.

---

## Objective 3 — Ereigniszentrierte Auswertung & direkter Klassifikator

**Zwei Richtungen:**

**3a — τ-Sweep:** Für alle Prognosemodelle wird die Zeittoleranz τ von 0 bis 12 Schritte (0–60 min) variiert. Ergebnis: Alle Prognosemodelle benötigen τ > 5 Schritte (25 min) für nicht-triviale Erkennungsrate. Bei der klinisch relevanten Toleranz τ=3 (15 min) versagen alle prognosebasierten Ansätze.

**3b — Direkter Binärklassifikator:** Statt den Glukoseverlauf vorherzusagen und Ereignisse durch Schwellenwertunterschreitung abzuleiten, wird direkt `p(Hypoglykämie innerhalb 60 min | Lookback)` vorhergesagt. Ausgabekopf: sigmoid, Verlustfunktion: BCE. Label: 1 wenn min(y_{t+0},...,y_{t+11}) < 70 mg/dL.

| Modell | Ansatz | Precision | Recall | F1 | F2 |
|--------|--------|-----------|--------|----|----|
| LSTM | Prognose-abgeleitet (Baseline MSE) | 0.010 | 0.004 | 0.005 | 0.004 |
| LSTM | Prognose-abgeleitet (Multi-step) | 0.011 | 0.009 | 0.010 | — |
| **LSTM** | **Direkter Binärklassifikator** | **0.237** | **0.693** | **0.334** | **0.463** |
| PatchTST | Prognose-abgeleitet (Baseline MSE) | 0.048 | 0.089 | 0.061 | 0.074 |
| PatchTST | Prognose-abgeleitet (Multi-step) | 0.060 | 0.103 | 0.072 | — |
| **PatchTST** | **Direkter Binärklassifikator** | **0.054** | **0.615** | **0.095** | **0.178** |

*F2 gewichtet Recall 4× stärker als Precision — relevant da verpasste Ereignisse klinisch gefährlicher sind als Fehlalarme.*

**Wichtigste Erkenntnisse:**

- **LSTM-Klassifikator: F1=0.334 — 62× Verbesserung** gegenüber prognoseabgeleitetem LSTM (F1=0.005). Recall=0.693 bedeutet 69% der Hypoglykämieereignisse werden innerhalb ±15 min erkannt.
- **PatchTST-Klassifikator: F1=0.095** — bescheidene Verbesserung (1.6×), sehr hohe Falschalarmrate (Precision=0.054).
- **Architektur-Inversion:** LSTM schlechter bei Prognose (RMSE 36.09 vs. 39.27), aber 3.5× bessere F1 bei direkter Klassifikation. Das rekurrente Gedächtnis des LSTM eignet sich besser für binäre Ereignisvorhersage; der PatchTST neigt dazu, jeden Abwärtstrend als potentielles Ereignis zu werten.
- Klassen-Ungleichgewicht bleibt ein fundamentales Problem: selbst beim LSTM-Klassifikator entspricht nur jede 4. positive Vorhersage einem echten Ereignis (Precision=0.237).

---

## Gesamtvergleich

| Modell | Verlust | RMSE | lag_rmse | Hypo-F1 |
|--------|---------|------|----------|---------|
| LSTM | MSE (Baseline) | 36.09 | — | 0.005 |
| LSTM | Bounded-Lag | 36.14 | 25.25 | 0.014 |
| LSTM | Soft-DTW | 37.23 | 28.67 | 0.014 |
| LSTM | Multi-step | **35.59** | **20.74** | 0.010 |
| LSTM | Direkter Klassif. | — | — | **0.334** |
| PatchTST | MSE (Baseline) | 39.27 | — | 0.061 |
| PatchTST | Bounded-Lag | 39.30 | 16.70 | 0.067 |
| PatchTST | Soft-DTW | 43.24 | 24.01 | 0.062 |
| PatchTST | Multi-step | **39.05** | **14.53** | 0.072 |
| PatchTST | Direkter Klassif. | — | — | 0.095 |

## Empfehlungen

| Aufgabe | Empfohlener Ansatz | Begründung |
|---------|-------------------|-----------|
| Glukoseverlauf-Prognose | LSTM Multi-step MSE | Bester RMSE + niedrigstes lag_rmse |
| Hypoglykämieerkennung | LSTM Binärklassifikator | 62× F1-Verbesserung, klinisch sinnvoller Recall |

**Kernbotschaft:** Prognose und Ereigniserkennung sollten als zwei separate Aufgaben behandelt werden — ein Multi-step-Prognosemodell für den Glukoseverlauf und ein separater binärer LSTM-Klassifikator für Hypoglykämiealarm. Dieser Ansatz entspricht dem Design von Hüni (2023).
