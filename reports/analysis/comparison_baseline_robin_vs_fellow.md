# Baseline-Vergleich: Robin van den Hoek vs. Mitstudierender (ahonongobi)

**Datum:** 2026-04-16  
**Branch (Robin):** `main`  
**Repo (Mitstudierender):** `git@github.com:ahonongobi/Master-Thesis.git`  
**Verglichene Notebooks:** `notebooks/02_lstm_baseline_final.ipynb`, `notebooks/train/task_502_lstm_val.ipynb`, `notebooks/train/task_502_patchtst_val.ipynb`

---

## 1. Executive Summary

Beide Implementierungen trainieren ein LSTM und ein PatchTST-Modell für 60-Minuten-Glukose-Forecasting
auf demselben 36-Patienten-Datensatz. Robin beobachtet einen klaren Time-Shift-Artefakt in seinen
Predictions — das Modell reproduziert die Glukosekurve mit zeitlicher Verzögerung. Der Mitstudierenden
berichtet keinen solchen Shift. **Der Shift ist beim Mitstudierenden jedoch nicht verschwunden, sondern
methodisch unsichtbar gemacht:** Sein RMSE wird über alle 12 Horizont-Steps gemittelt (5–60 min), und
seine Plots vergleichen die 5-Minuten-Prediction mit der 5-Minuten-Wahrheit — einem Horizont, bei dem
kaum Shift entsteht. Dazu kommen drei weitere schwerwiegende methodische Fehler: Data Leakage in der
Normalisierung, Überschreitung von Patienten-Grenzen in Sliding Windows, und ein einzelnes globales
Modell für alle 36 Patienten. Die gemeldeten Metriken der beiden Ansätze sind daher nicht direkt
vergleichbar.

---

## 2. Überblick aller Unterschiede

| Aspekt | Robin (main) | Mitstudierender |
|---|---|---|
| **Evaluation-Ziel** | **Nur h=11 (60-min Step)** | **Durchschnitt über Steps 1–12** |
| **Plot zeigt** | **60-min Prediction** | **5-min Prediction (Step 0)** |
| Normalisierung | Per-Patient Z-Score (nur Train) | Global MinMaxScaler (ganzer Datensatz) |
| Patient-Grenzen | Respektiert (per-patient Dataset) | Nicht respektiert (konkateniert) |
| Modell-Typ | 36 Patient-spez. Modelle | 1 globales Modell |
| Model output (horizon) | 1 (direkter 60-min Output) | 12 (alle Steps) |
| Training-Loss Ziel | MSE auf y[:,11] (60-min) | MSE/Huber über alle y[:,0:12] |
| Lookback | 24 Steps = 2h | 72 Steps = 6h |
| Loss-Funktion | MSE | SmoothL1 (Huber) |
| Hyperparameter-Tuning | Optuna (50 Trials) | Fix gesetzt |
| Train/Val/Test Split | 60/20/20 temporal, pro Patient | 70/15/15 temporal, gepooled |

---

## 3. Detailanalyse

---

### 3.1 Evaluation-Scope — Hauptursache des fehlenden Shifts

Dies ist der zentrale Unterschied, der erklärt warum der Mitstudierenden keinen Shift sieht.

#### Robin

Das Modell hat `horizon=1` (gibt nur einen Wert aus). Evaluation und RMSE beziehen sich ausschliesslich
auf den 60-Minuten-Step:

```python
# scripts/train_lstm_60min.py, eval_hstep_trace()
return ys[:, h_index], yhat[:, 0]   # h_index=11 → 60-min Step
```
```python
# training loss:
loss = loss_fn(model(x)[:, 0], y[:, h_index])   # nur auf 60-min Step
```

Der gemeldete RMSE = **36.09 mg/dL** ist ein reiner 60-Minuten-Wert.

#### Mitstudierender (task_502_lstm_val.ipynb)

Das Modell hat `output_size=HORIZON=12`. Die `create_sequences`-Funktion erstellt y-Fenster mit 12 Steps.
RMSE wird über ALLE 12 Steps gemittelt:

```python
# create_sequences: y enthält 12 Steps
y.append(data[i + seq_len : i + seq_len + horizon])   # Shape pro Sample: (12,)

# Evaluation (Cell 6):
y_pred = scaler.inverse_transform(y_pred_scaled)   # Shape: (N, 12)
y_true = scaler.inverse_transform(y_true_scaled)   # Shape: (N, 12)
rmse = np.sqrt(np.mean((y_pred - y_true)**2))      # ALLE 12 Steps gemittelt
```

Der Plot (Cell 9) zeigt ausserdem **Step 0** (5-min-ahead), nicht Step 11 (60-min):

```python
plt.plot(y_true[:200, 0], label="Actual Glucose")    # Step 0 = t+5 min
plt.plot(y_pred[:200, 0], label="LSTM Prediction")   # Step 0 = t+5 min
```

#### Warum der Shift dadurch unsichtbar wird

Der Time-Shift-Artefakt ist ein Horizon-abhängiges Phänomen: Er wächst mit der Vorhersagezeit.
Bei 5-Minuten-ahead ist der Shift minimal (das Modell kann nahezu gut extrapolieren), bei
60-Minuten-ahead ist er maximal (das Modell tendiert zur Wiedergabe des letzten bekannten Werts).

- **Gemittelter RMSE:** Die Steps 1–6 (5–30 min) haben deutlich niedrigeren Fehler als Step 12 (60 min).
  Der Durchschnitt wird von den kurzen Horizonten dominiert. Der 60-min-Fehler wird verwässert.
- **Plot bei Step 0:** Bei t+5 min sieht die Prediction immer gut aus. Kein Shift sichtbar.
- **Schlussfolgerung:** Der Shift ist in den Daten vorhanden, wird aber nie gemessen oder visualisiert.
  Würde man `y_pred[:, 11]` vs. `y_true[:, 11]` plotten, würde man denselben Shift wie bei Robin sehen.

#### Metriken-Vergleich — nicht vergleichbar

| Modell | RMSE (gemeldet) | Was RMSE bedeutet |
|---|---|---|
| Robin LSTM | 36.09 mg/dL | Reiner 60-min RMSE |
| Fellow LSTM | 27.61 mg/dL | Durchschnitt Steps 1–12 (5–60 min) |
| Robin PatchTST | 39.27 mg/dL | Reiner 60-min RMSE |
| Fellow PatchTST | 23.51 mg/dL | Durchschnitt Steps 1–12 (5–60 min) |

Die scheinbar besseren Metriken des Mitstudierenden sind methodisch nicht vergleichbar.
Sie messen ein leichteres Problem (gemischte Kurzzeit-Langzeit-Accuracy) statt des eigentlichen
Ziels (60-Minuten-Forecasting).

---

### 3.2 Normalisierung mit Data Leakage

#### Robin

Per-Patient Z-Score-Normalisierung. Der Scaler wird ausschliesslich auf dem **Train-Split** des
jeweiligen Patienten gefittet und dann auf Val/Test angewendet:

```python
# src/ba_baseline/data/multi_patient_dataset.py
mean = float(train_s.mean())   # nur Train-Daten
std  = float(train_s.std())

# Normalisierung:
x = (x - self.mean) / (self.std + 1e-8)
y = (y - self.mean) / (self.std + 1e-8)
```

Kein Data Leakage: zukünftige Test-Werte haben keinen Einfluss auf die Normalisierung.

#### Mitstudierender

Ein globaler `MinMaxScaler` wird auf **den gesamten Datensatz** gefittet (alle Patienten, alle Splits):

```python
# task_502_lstm_val.ipynb, Cell 3:
glucose = data['glucose_level'].values.reshape(-1, 1)   # 314194 Punkte = ganzer Datensatz
scaler = MinMaxScaler()
glucose_scaled = scaler.fit_transform(glucose)   # FIT auf Train + Val + Test!

# Danach erst Split:
n = len(X)
train_end = int(0.7 * n)
val_end   = int(0.85 * n)
```

**Problem:** Der globale Min/Max-Wert (z.B. niedrigster/höchster Glukosewert irgendeines Patienten
im Testset) fliesst in die Normalisierung aller Daten ein. Das Modell "weiss" dadurch indirekt über
den Wertebereich zukünftiger Daten Bescheid — ein klassisches Data-Leakage-Muster.

Zusätzlich: Eine globale Normalisierung über alle Patienten ignoriert die unterschiedlichen
physiologischen Grundniveaus der 36 Patienten. Patient-spezifische Glukosemuster werden nivelliert.

---

### 3.3 Kreuzung von Patienten-Grenzen in Sliding Windows

#### Robin

Das `MultiPatientWindowDataset` iteriert pro Patient separat. Sliding Windows bleiben innerhalb
der Zeitreihe eines einzelnen Patienten:

```python
# src/ba_baseline/data/multi_patient_dataset.py
for pid in patient_ids:
    s = np.asarray(series_by_patient[pid], dtype=np.float32)
    max_start = len(s) - (self.lookback + self.horizon)
    for i in range(max_start):
        self.index.append((pid, i))   # nur innerhalb dieses Patienten
```

Jedes Trainingsfenster gehört genau einem Patienten. Keine Grenz-Überschreitung.

#### Mitstudierender

Alle 36 Patienten werden konkateniert, dann werden Sliding Windows über die gesamte Sequenz gezogen:

```python
# task_502_lstm_val.ipynb, Cell 3:
glucose = data['glucose_level'].values.reshape(-1, 1)   # alle Patienten hintereinander

def create_sequences(data, seq_len, horizon):
    for i in range(len(data) - seq_len - horizon):
        X.append(data[i : i + seq_len])   # kann Grenze zwischen Patienten überschreiten!
```

**Problem:** An den Übergängen zwischen Patienten (z.B. letzter Wert von Patient 85101 → erster Wert
von Patient 85102) entstehen Fenster, die physiologisch zusammenhangslose Daten kombinieren. Das Modell
wird auf unechten Sequenzen trainiert. Da es 36 Patienten gibt mit je ca. 8700 Punkten im Schnitt,
entstehen an den 35 Grenzen je bis zu `seq_len-1 = 71` ungültige Trainingsfenster (insgesamt ~2500
korrupte Fenster). Bei dem 3.78 Mio Samples in `task_502_patchtst_val.ipynb` ist der Einfluss gering
aber methodisch nicht korrekt.

---

## 4. Schlussfolgerung

Der Time-Shift-Artefakt ist ein intrinsisches Phänomen von direktem MSE-Training auf langen Horizonten.
Er tritt bei beiden Implementierungen auf. Beim Mitstudierenden ist er methodisch unsichtbar, weil:

1. **Die Evaluation und der Plot beziehen sich auf Step 0 (5-min) statt Step 11 (60-min).** Bei
   kurzen Horizonten ist der Shift vernachlässigbar.
2. **Der gemeldete RMSE ist der Durchschnitt über alle 12 Steps** — damit ist er mit Robins 60-min RMSE
   weder vergleichbar noch interpretierbar als reine 60-Minuten-Genauigkeit.

Dazu kommen drei weitere methodische Probleme (Data Leakage, Patienten-Grenzen, globales Modell), die
die Ergebnisse des Mitstudierenden zusätzlich von einer seriösen Baseline unterscheiden.

**Robins Implementierung** entspricht dem Proposal (direktes 60-min Forecasting, per-patient Modelle,
sauberer temporaler Split, kein Leakage) und misst das richtige Problem. Der beobachtete Shift ist
ein echter und valider Befund, der die Motivation für die Erweiterungsobjektive (Objectives 1–3)
direkt begründet.

---

## 5. Gemeldete Metriken im Kontext

| Modell | RMSE | Methode | Vergleichbar? |
|---|---|---|---|
| **Robin LSTM** | **36.09 mg/dL** | **60-min RMSE, per-patient, kein Leakage** | Referenz |
| **Robin PatchTST** | **39.27 mg/dL** | **60-min RMSE, per-patient, kein Leakage** | Referenz |
| Fellow LSTM (alt) | 37.44 mg/dL | 60-min Punkt, global, Leakage | Bedingt |
| Fellow LSTM (neu) | 27.61 mg/dL | Ø Steps 1–12, global, Leakage | **Nicht vergleichbar** |
| Fellow PatchTST | 23.51 mg/dL | Ø Steps 1–12, global, Leakage | **Nicht vergleichbar** |

Die alte Version des Mitstudierenden (`02_lstm_baseline_final.ipynb`, RMSE = 37.44 mg/dL) ist die
einzig annähernd vergleichbare Zahl: Dort wurde ebenfalls ein einzelner 60-min-Wert als Target verwendet.
Die Differenz zu Robins 36.09 mg/dL ist erklärbar durch globales Modell, globale Normalisierung und
Huber-Loss statt MSE.
