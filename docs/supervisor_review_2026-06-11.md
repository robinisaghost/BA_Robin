# Supervisor-Review der Bachelorarbeit
**Titel:** *Mitigating Time-Shift Errors in CGM-based Glucose Forecasting and Hypoglycemia Event Prediction*
**Autor:** Robin van den Hoek · **Prüfmassstab:** Riesen, *From Research to Thesis* (v2.0) + AWL
**Datum:** 2026-06-11

---

## 0. Gesamturteil

Inhaltlich ist die Arbeit stark: klare Forschungsfragen als kontrollierte Ablation, ein
durchgehender roter Faden (Time-Shift → drei Mitigationsstrategien → Trennung von
Forecasting und Detection), und mehrere echte „eine Ebene tiefer"-Analysen
(RMSE* zur Trennung von Timing und Form; Lead-Time-Analyse; Architektur-Inversion).
Das entspricht genau dem, was die Vorgaben unter *Scientific Argumentation* und
*Measure and Analyze at Least one Level Deeper* fordern.

Zwei Dinge müssen vor der Abgabe zwingend adressiert werden:

1. **Länge.** Einleitung–Conclusion umfasst aktuell **43 Seiten** (Soll ~30, max. 35).
   Das ist 8–13 Seiten über dem Ziel. → Abschnitt 1.
2. **Datensatz-Quelle (`garciatirado2023t1d`).** Die zitierte Studie beschreibt
   einen **stationären Crossover-Trial mit drei 24-Stunden-Klinikaufenthalten** und
   **35 ausgewerteten** Teilnehmenden (36 randomisiert, 1 ausgeschlossen) — nicht
   „ca. 30 Tage pro Patient" und nicht „36 Patienten", wie die Arbeit durchgehend
   schreibt. → Abschnitt 2.1 (kritisch).

---

## 1. Seitenlänge (höchste Priorität)

Ist-Zustand laut `main.toc`:

| Kapitel | Seiten | Umfang | Soll (30-S.-Arbeit) |
|---|---|---|---|
| 1 Introduction | 1–4 | 4 S. | ~3 S. (10 %) |
| 2 Theory and Background | 5–16 | **12 S.** | — |
| 3 Experimental Design | 17–22 | 6 S. | (Kap. 2–4 zus. ~24 S.) |
| 4 Empirical Evaluation | 23–38 | **16 S.** | — |
| 5 Conclusions | 39–43 | 5 S. | ~3 S. (10 %) |
| **Intro–Conclusion** | **1–43** | **43 S.** | **30, max 35** |

Der Überhang sitzt im Hauptteil, vor allem in **Kap. 4 (16 S.)** und **Kap. 2 (12 S.)**.

**Hebel 1 — Redundanz beim Time-Shift-Mechanismus entfernen (~3–4 S.).**
Die Kette „autokorreliertes Signal → MSE belohnt Kopieren der Vergangenheit → Delay"
wird mindestens fünfmal nahezu identisch erklärt: Abstract, Einleitung (Abs. 3),
Background 2.2.1 (MSE), Methode 3.2.1 (Autokorrelation), Baseline 4.1, Conclusion.
Die Vorgaben erlauben bewusste Wiederholung *nur in anderer Form* und warnen
ausdrücklich vor „uncontrolled redundancy". Empfehlung: Mechanismus **einmal** formal
im Background herleiten, an den übrigen Stellen mit einem Satz + Verweis darauf
referenzieren.

**Hebel 2 — Kap. 4 straffen (~4–5 S.).** Pro Objective stehen mehrere lange
Interpretationsabsätze, gefolgt von „Overall Comparison" (4 Absätze), das vieles
nochmals zusammenfasst. Die Section 4.5 überschneidet sich stark mit den
Einzel-Objectives und mit der Conclusion. Empfehlung: 4.5 auf ~1 Absatz +
Übersichtstabelle kürzen; pro Objective je einen Interpretationsabsatz streichen.
Beobachtung und Interpretation lassen sich oft in einem Satz bündeln.

**Hebel 3 — Kap. 2 verdichten (~2–3 S.).** Die LSTM-Gleichungen samt Fliesstext,
die volle Soft-DTW-Herleitung und die Metrik-Abschnitte sind didaktisch gut, aber
ausführlich. Die sehr langen Sub-Caption-Texte in Fig. 2.x (Loss-Geometrie) lassen
sich um die Hälfte kürzen. Standardstoff (LSTM-Gating) darf knapper sein — die
Vorgabe sagt: Werkzeug-Konzepte knapp, nur das thesis-spezifische ausführlich.

**Hebel 4 — Conclusion (~1 S.).** Die *Summary of Findings* referiert die Resultate
fast vollständig nochmals. Die Vorgabe verlangt, dass die Conclusion sich inhaltlich
von Abstract/Resultaten unterscheidet und **knapper** ist. Antworten auf RQ1–3 je auf
3–4 Sätze kürzen.

Realistisch sind so 9–12 Seiten einsparbar, womit die Arbeit im 31–34-S.-Korridor landet.

---

## 2. Referenzprüfung

Methodik: Jede Kernquelle gegen Original (DOI/Verlag/PubMed/arXiv) geprüft —
Existenz, korrekte bibliografische Angaben und ob der **Verwendungszweck im Text**
durch die Quelle gedeckt ist.

### 2.1 Kritisch — Datensatzbeschreibung deckt sich nicht mit der Quelle

`garciatirado2023t1d` (Garcia-Tirado et al., *Diabetes Care* 46(9):1652–1658, 2023 —
bibliografisch korrekt, DOI stimmt). **Aber** die Original-Studie ist ein
**randomisierter Crossover-Trial mit drei aufeinanderfolgenden 24-Stunden-
Klinikaufenthalten**; ausgewertet wurden **35** Erwachsene (36 randomisiert, einer
nachträglich ausgeschlossen). Gerät Dexcom G6 ✓ und „randomised crossover" ✓ stimmen.

Die Arbeit schreibt dagegen in 3.2.1 (und im Abstract/Intro):
> „36 anonymised patients recorded at 5-minute intervals over **approximately 30 days
> per patient**".

Diskrepanzen, die der Prüfer mit Sicherheit anspricht:
- **Dauer:** „~30 Tage/Patient" vs. drei 24-h-Aufenthalte in der Publikation.
- **N:** „36" vs. 35 ausgewertete in der Publikation.

→ **Mit den tatsächlichen Daten abgleichen.** Entweder nutzt du eine erweiterte/
ambulante Daten-Release dieses Trials (dann muss die Beschreibung genau diese Quelle/
Release nennen und die 30 Tage belegen), oder die Beschreibung ist falsch und muss
korrigiert werden. Falls 36 ≠ 35 bewusst (alle randomisierten inkl. Ausgeschlossenem),
in einer Fussnote begründen. Dies ist kein Stilpunkt, sondern Reproduzierbarkeit
(Vorgabe Kap. 4.1: alle Datenquellen reproduzierbar beschreiben).

### 2.2 Präzision der Verwendung

`cinar2025hypoclassification` (arXiv 2504.00009 — Titel/Autoren korrekt). Wird in
Einleitung **und** 4.4.2 zusammen mit `shao2024` als Beleg dafür zitiert, dass
„end-to-end binary classifiers **outperform threshold-based regression detection**".
Die Cinar-Arbeit vergleicht jedoch laut Abstract ResNet vs. LSTM und
subjekt- vs. populationsspezifische Modelle über mehrere Horizonte — **keinen**
Vergleich gegen schwellenbasierte Regressionsdetektion. → Den *komparativen* Anspruch
auf `shao2024` (bzw. eine Quelle, die diesen Vergleich wirklich zieht) stützen und
`cinar2025` nur für „Machbarkeit/Architekturwahl der Klassifikation über Horizonte"
verwenden. (Vorgabe: *Cite with Purpose* — Beleg muss den behaupteten Inhalt decken.)

### 2.3 Verifiziert und korrekt verwendet

| Schlüssel | Status | Verwendung gedeckt? |
|---|---|---|
| `wolff2025cgpm` | DTT 27(10):858–870, 2025, DOI ✓ | Ja — RMSE belohnt triviale Last-Value-Prädiktion ✓ |
| `fox2018multioutput` | KDD 2018 ✓ | Ja — Multi-Output-Supervision verbessert Genauigkeit ✓ |
| `shao2024hypoglycemia` | JMIR Med Inform 2024, DOI ✓ | Ja — CGM-Hypo-Klassifikation ✓ |
| `sun2026futureaure` → `sun2026futureaware` | Sci Rep 16:11404, 2026, DOI ✓ | Ja — future-aware/privileged-info Training ✓ (Datum echt 2026) |
| `hochreiter1997lstm`, `bengio1994vanishing`, `vaswani2017attention` | Klassiker ✓ | Ja |
| `nie2023patchtst`, `kim2022revin`, `cuturi2017softdtw` | ✓ | Ja |
| `sakoe1978dtw`, `berndt1994dtw` | ✓ | Ja |
| `dimeglio2018t1d`, `bekiari2018aid`, `battelino2019` | ✓ | Ja |
| `idf2021atlas` | 537 Mio. (2021) ✓ | inhaltlich korrekt; s. 2.4 zur Quellenwahl |
| `hovorka2011closedloop` | Nat Rev Endocrinol 7:385–395 ✓ | „15–45 min Wirkverzögerung" plausibel; Zahl gegen Volltext gegenprüfen |

### 2.4 Bibliografie-Hygiene

- **`\nocite{*}` in `main.tex` (Z. 94) entfernen.** Es zwingt **alle** `lib.bib`-Einträge
  in die Bibliografie — auch nicht zitierte. Konkret erscheint `velickovic2018gat`
  (Graph Attention Networks), das im Text **nirgends** zitiert wird (GAT wird in der
  Einleitung nur im Zusammenhang mit Hünis Arbeit erwähnt, ohne `\citep`). Vorgabe:
  Bibliografie = zitierte Werke. → entweder `velickovic2018gat` an der GAT-Stelle
  zitieren oder Eintrag löschen; `\nocite{*}` in jedem Fall raus.
- **`idf2021atlas`** ist eine Website (`@misc`). Die Vorgabe rät, Websites zu meiden und
  peer-reviewte Quellen vorzuziehen. Für die 537-Mio.-Zahl existiert der referierte
  Artikel (Sun et al., *Diabetes Res Clin Pract* 183:109119, 2022, „IDF Diabetes Atlas …
  estimates for 2021"). → ersetzen.
- **`prg_glucose_proposal`** (unpubliziert, intern) trägt zentrale Methoden
  (Bounded-Lag-Loss, RMSE*, τ-Fenster). Für eine Gruppen-Arbeit vertretbar, aber mit
  dem Betreuer klären, ob/wie zitierbar; ggf. die Methoden eigenständig herleiten.
- **`huni2023glucose`** trägt im `.bib` noch den Kommentar „TODO: confirm full reference".
  → vor Abgabe auflösen.
- Anzahl zitierter Quellen (~27) passt zur Heuristik (~1 Quelle/Seite). ✓

### 2.5 Vergleichbarkeit der F2-Werte (methodischer Hinweis, kein Zitatfehler)

Abstract/Conclusion stellen „Klassifikator-F2 0.46 vs. forecast-derived F2 < 0.09"
gegenüber. Beachte: Die direkten Klassifikatoren werden **pro Fenster** ausgewertet,
die forecast-derived Detektoren über **Crossing-Matching innerhalb τ** (Background 2.3.2).
Das sind **unterschiedliche Auswertungsprotokolle** — die Schlagzeile vergleicht also
nicht ganz gleichartig. Zudem erreicht der forecast-derived PatchTST bei τ=60 ein F2 von
**0.64** (Anhangstabelle), also über dem LSTM-Klassifikator (0.46). → In Abstract/
Conclusion klarstellen, dass der Klassifikator-Vorteil bei **klinisch relevantem τ=15**
gilt und dass die beiden F2 unter verschiedenen Protokollen berechnet sind. Das nimmt
einem kritischen Prüfer den stärksten Einwand vorweg.

---

## 3. Kapitelweises Review

### Abstract
- Länge (~250 Wörter) und Zitatfreiheit ✓ (Vorgabe erfüllt). Self-contained ✓.
- „36 adults" → s. 2.1.
- Sehr dicht; gut. Ein Nebensatz, dass die Klassifikator- vs. Forecast-F2 unter
  verschiedenen Protokollen stehen, würde Überinterpretation vorbeugen (2.5).

### Kapitel 1 — Introduction
- Top-down-Aufbau (T1D → CGM → Hypoglykämie → Time-Shift → Lücke → Fragen → Beiträge)
  entspricht der Vorgabe genau; Research Gap und Goals sind explizit. Sehr gut.
- 4 Seiten; bei 30-S.-Ziel auf ~3 kürzen (Hebel 1).
- Der Related-Work-Block (Abs. 5–6) ist eigentlich *Related Work* und gehört teils in
  Kap. 2; aktuell verdoppelt er sich mit Background. Prüfen, ob nicht eine der beiden
  Stellen genügt (spart Platz, vermeidet Redundanz).
- „AID systems" — automated insulin delivery wird ausgeschrieben, gut; Abkürzung wird
  nicht weiter gebraucht, also korrekt nicht eingeführt.

### Kapitel 2 — Theory and Background
- Strukturkonsistenz (je ≥2 Subsections) ✓.
- Titel „Theory and Background" ist etwas generisch; akzeptabel.
- 2.2.1 MSE: enthält die zweite vollständige Delay-Herleitung — als *die* formale
  Stelle behalten, dafür Intro/Baseline/Conclusion dort entlasten (Hebel 1).
- Soft-DTW (2.2.2): sehr gründlich und korrekt; für eine BSc-Arbeit eher lang. Ein
  Absatz lässt sich streichen.
- Fig. 2.x Loss-Geometrie: Sub-Captions sind ganze Absätze. Vorgabe „A Caption Is Not a
  Decoration" verlangt aussagekräftige, aber knappe Captions → ~50 % kürzen.
- Konsistente Nutzung von „Fig."/„Tab." ✓, nummerierte Vancouver-Zitate ✓.

### Kapitel 3 — Experimental Design and Training Objectives
- Klar und reproduzierbar aufgebaut; Problemformalisierung sauber.
- **3.2.1:** Datensatzbeschreibung korrigieren/belegen (2.1). „T1DATA" ist deine eigene
  Bezeichnung für den Garcia-Tirado-Datensatz — beim ersten Auftreten explizit so
  einführen („im Folgenden *T1DATA*"), sonst liest es sich wie ein etablierter
  Benchmark.
- **Statistik:** Es werden Kohorten-**Mittelwerte** über 36 Patienten verglichen, aber
  (ausser den F2-SD der Klassifikatoren) ohne Streuung/Konfidenz oder Signifikanztest.
  Aussagen wie „considerably higher F2" wären mit einem gepaarten Test (z. B. Wilcoxon
  signed-rank über Patienten) deutlich belastbarer. Die Vorgabe (Kap. 4.1/4.2) verlangt
  Tests, *wenn* Signifikanz behauptet wird — du vermeidest das Wort sauber, aber ein
  Test würde die zentralen Vergleiche absichern. Empfehlung: für die Hauptvergleiche
  (Klassifikator vs. bester forecast-derived; Multi-Step vs. Baseline) je einen
  gepaarten Test ergänzen.
- Tab. 3.1 Hyperparameter: vorbildlich (Tuning-Patient genannt, Limitation verlinkt).

### Kapitel 4 — Empirical Evaluation
- Description/Interpretation werden meist sauber getrennt (Vorgabe 4.6) — gut.
- Straffen (Hebel 2). Insbesondere 4.5 „Overall Comparison" überschneidet sich mit den
  Objectives und mit Kap. 5.
- 4.4.2: Die RevIN-Erklärung für die niedrige PatchTST-Präzision ist eine gut
  argumentierte Claim-Evidence-Reasoning-Kette mit anschliessender Limitation — genau
  im Sinne der Vorgaben. Behalten.
- Tabellen: Mittelwerte ohne Streuung (s. o.). Mindestens für die Kernvergleiche SD
  oder CI ergänzen.
- Bold-„best per column" ist konsistent definiert ✓.

### Kapitel 5 — Conclusions and Future Work
- Beantwortet RQ1–3 explizit und zieht eine übergreifende Synthese — strukturell genau
  richtig.
- Zu lang/zu nah an den Resultaten (Hebel 4) — kürzen, damit es sich vom Resultatkapitel
  abhebt.
- Limitationen (univariat, Zero-Lead-Time, Imbalance, Single-Patient-Tuning) sind ehrlich
  und konkret — sehr gut (Vorgabe: Limitationen stärken die Arbeit).
- Future Work ist konkret und aus den Limitationen abgeleitet ✓.

### Anhang
- Sinnvoll ausgelagert (τ-Sweep-Tabellen, Konfusionszählungen, PR-Raum). ✓
- `pr_paradox.png` / Fig. A.2 wird im Haupttext nicht referenziert — entweder im Text
  per `\ref` einbinden (Vorgabe: jede Abbildung wird im Fliesstext referenziert) oder
  entfernen.

---

## 4. Sprache & Form (gegen die Checklisten)

Insgesamt sehr nah an den Vorgaben:
- Unpersönlicher Stil, kein „I/we", keine Kontraktionen, keine rhetorischen Fragen ✓.
- „significant" wird vermieden zugunsten von „considerably/markedly/substantially" ✓.
- Tausendertrennzeichen mit Komma (`57{,}680`) ✓; Präsens konsistent ✓.
- Abkürzungen bei Erstnennung eingeführt (CGM, MSE, RevIN, DTW, LSTM, RMSE*) ✓.

Kleinere Punkte:
- Fig. 2.2 (PatchTST): Caption „Source: Nie et al." — die Abbildung ist laut Text „adapted
  to the CGM setting". Wenn nachgezeichnet/angepasst: „**Adapted from**" statt „Source:".
- Bullet-Listen (Beiträge, Objectives): grammatisch parallel ✓.

---

## 5. Front-/Back-Matter

- **Eigenständigkeitserklärung fehlt.** `main.tex` bindet keine „Erklärung" ein. Die
  Vorgabe verlangt die unterschriebene Erklärung am Ende der Arbeit (philnat-Formular).
  → ergänzen.
- **Acknowledgements** stehen noch in der `planned`-Box (Platzhalter-Umgebung) — in
  echten Fliesstext umwandeln, `planned`-Boxen vor Abgabe überall entfernen.
- **Titelseite:** `\thesisauthororigin` = „[City of birth, Country]" ist ungenutzter
  Platzhalter; Seite zeigt „from Bern, Switzerland" — Platzhalter entfernen. Titel ist
  beschreibend ✓, aber lang (umbricht); ein knapperer Haupttitel + Untertitel wäre
  konform zur Vorgabe „Titel max. eine Zeile".

---

## 6. Priorisierte Massnahmenliste

**Muss vor Abgabe (inhaltlich/kritisch):**
1. Datensatzbeschreibung 3.2.1 mit `garciatirado2023t1d` in Einklang bringen (Dauer, N) —
   oder korrekte Quelle/Release nennen. (2.1)
2. Länge auf ≤35 S. bringen (Redundanz-Abbau + Kap. 4/2 straffen). (1)
3. `\nocite{*}` entfernen; `velickovic2018gat` zitieren oder löschen. (2.4)
4. Komparativen Anspruch von `cinar2025` auf eine deckende Quelle umhängen. (2.2)
5. Eigenständigkeitserklärung einfügen; Acknowledgements/`planned`-Boxen finalisieren. (5)

**Sollte (Qualität/Robustheit):**
6. Gepaarte Signifikanztests + Streuung für die Kernvergleiche. (3, 4)
7. F2-Vergleich Klassifikator vs. forecast-derived als protokoll-/τ-abhängig klarstellen. (2.5)
8. `idf2021atlas` durch peer-reviewte Quelle ersetzen; `huni2023`-TODO auflösen. (2.4)
9. Fig. A.2 im Text referenzieren oder entfernen; lange Captions kürzen. (3-Anhang, 2)

**Kann (Feinschliff):**
10. „T1DATA" als eigene Bezeichnung explizit einführen; „Source:"→„Adapted from"; Titel kürzen.
