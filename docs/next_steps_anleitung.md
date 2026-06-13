# Anleitung für den nächsten Arbeitsschritt
**Ziel:** Referenzen vollständig bereinigen, Länge auf ≤35 Seiten bringen, Datensatz-
Quelle korrekt machen, Eigenständigkeitserklärung + Acknowledgements ergänzen, alle
Anhang-Teile im Text referenzieren.
**Nicht im Auftrag (bewusst ausgeschlossen):** Streuungsmaße und Signifikanztests —
wurden nicht mit dem Supervisor besprochen und werden **nicht** eingebaut.
**Stand:** 2026-06-11 · Basis: Review in `docs/supervisor_review_2026-06-11.md`

---

## Arbeitspaket A — Datensatz-Quelle korrigieren (T1DATA)

**Verifizierte Fakten (aus `data/raw/all_cgm.csv`, nicht ändern):**
- **36 Patienten** (Spalte `patient_ID`, 36 eindeutige IDs)
- **314.194 Messwerte** gesamt → ~8.728 pro Patient → **~30 Tage pro Patient** bei 5-min-Takt
- Dexcom G6, 5-Minuten-Intervall

→ Die Zahlen in der Thesis (36 Patienten, ~30 Tage) sind **korrekt** und bleiben.
Das Problem ist die Beschreibung als „randomised crossover clinical trial" mit Verweis
auf Garcia-Tirado 2023 — deren Trial sind drei stationäre 24-h-Aufenthalte mit 35
ausgewerteten Personen, was **nicht** zu einem 30-Tage-Datensatz passt.

**A1 — Herkunft bestätigen (eine Entscheidung nötig).**
Kläre, woher `all_cgm.csv` stammt. Zwei mögliche Auflösungen:
- *(a)* Es ist eine **andere/erweiterte Daten-Release** desselben Forschungs­umfelds
  (UVA/Breton-Gruppe) mit ~30 Tagen ambulanter CGM-Aufzeichnung. → Dann die **Beschreibung
  an die tatsächlichen Dateneigenschaften anpassen** (s. A2) und die korrekte Quelle
  dieser Release zitieren (mit Supervisor bestätigen).
- *(b)* Es ist genau die Garcia-Tirado-Kohorte, aber mit der vollständigen ambulanten
  Sensor-Tragedauer (nicht nur die supervidierten Aufenthalte). → Dann Zitat behalten,
  aber die **Protokoll-Formulierung entschärfen** (s. A2).

In beiden Fällen gilt: Die Thesis darf **keine Studien-Details behaupten, die der Quelle
widersprechen**. Falls die Herkunft nicht eindeutig belegbar ist, beschreibe die Daten
neutral nach ihren Eigenschaften und nenne den Bezug zur Garcia-Tirado-Kohorte als
„abgeleitet von / bereitgestellt durch die PRG" — mit Supervisor abstimmen.

**A2 — `thesis/method.tex`, Abschnitt 3.2.1 (`sec:method:setup:cgm`) umformulieren.**
Aktuell: „a randomised crossover clinical trial in which adults … used Dexcom G6 …
36 anonymised patients … over approximately 30 days per patient."
Neu (an Datenfakten ausrichten, ohne widersprüchliche Trial-Details):
- „T1DATA" beim ersten Auftreten ausdrücklich als **eigene Bezeichnung** einführen,
  z. B.: „… the dataset referred to here as *T1DATA*, comprising continuous outpatient
  CGM recordings from **36** adults with Type 1 diabetes (Dexcom G6, 5-minute sampling,
  approximately **30 days** per patient, **314{,}194** glucose readings in total)."
- Die Formulierung „randomised crossover clinical trial" nur beibehalten, wenn A1(b)
  bestätigt ist; sonst streichen.
- Quelle entsprechend A1 zitieren.

**A3 — Code-Docstring korrigieren:** `src/ba_baseline/data/patient_loader.py`, Zeilen 6–8:
„35 individuals" → **„36 individuals"**, damit Code, Daten und Thesis übereinstimmen.
`REFERENCES.md` [9] ggf. an die in A1 bestätigte Quelle anpassen.

**Akzeptanzkriterium:** Thesis, `patient_loader.py` und `REFERENCES.md` nennen
einheitlich 36 Patienten / ~30 Tage; keine Aussage widerspricht der zitierten Quelle.

---

## Arbeitspaket B — Referenzen bereinigen

**B1 — `\nocite{*}` entfernen.** `thesis/main.tex`, Zeile 94 (`\nocite{*}`) löschen.
Wirkung: Es erscheinen nur noch tatsächlich zitierte Quellen in der Bibliografie.

**B2 — `velickovic2018gat` (Graph Attention Networks) verarzten.**
Wird im Text nirgends zitiert (taucht nur über `\nocite{*}` auf). Zwei Optionen:
- **Bevorzugt:** in `thesis/introduction.tex` an der Stelle zitieren, wo Hünis Arbeit
  „LSTM and Graph Attention Networks" erwähnt → `Graph Attention Networks~\citep{velickovic2018gat}`.
- Alternativ: Eintrag aus `thesis/lib.bib` löschen.
(Nach B1 ist genau eine der beiden Optionen nötig, damit keine unzitierte Quelle bleibt.)

**B3 — `cinar2025hypoclassification` korrekt einsetzen.**
Der komparative Anspruch „binary classifiers **outperform threshold-based regression
detection**" wird von der Cinar-Arbeit **nicht** belegt (sie vergleicht ResNet vs. LSTM
über Horizonte). Betroffene Stellen anpassen:
- `thesis/introduction.tex` (Related-Work-Absatz, „\citet{shao2024…} and
  \citet{cinar2025…} show that end-to-end binary classifiers outperform threshold-based
  regression detection") → den *outperform*-Anspruch nur auf `shao2024` stützen;
  `cinar2025` umformulieren zu „… and binary hypoglycaemia classification across multiple
  prediction horizons has been studied by~\citet{cinar2025hypoclassification}."
- `thesis/background.tex` 2.2.3 und `thesis/experiments.tex` 4.4.2: gleiche Trennung —
  `cinar2025` nur als Beleg für „Machbarkeit der Klassifikation", nicht für den Vergleich.

**B4 — `idf2021atlas` durch peer-reviewte Quelle ersetzen.**
In `thesis/lib.bib` den `@misc`-Website-Eintrag ersetzen durch den referierten Artikel:
Sun H. et al. (2022), „IDF Diabetes Atlas: Global, regional and country-level diabetes
prevalence estimates for 2021 and projections for 2045", *Diabetes Research and Clinical
Practice* 183:109119, DOI 10.1016/j.diabres.2021.109119. `\citep`-Schlüssel im Text
beibehalten oder konsistent umbenennen.

**B5 — `huni2023glucose` finalisieren.** Den `.bib`-Kommentar „TODO: confirm full
reference with supervisor" auflösen: Titel/Jahr/Typ mit dem Supervisor bestätigen,
TODO-Kommentar entfernen.

**B6 — (niedrige Priorität) `hovorka2011closedloop`.** Die Zahl „15 to 45 minutes" gegen
den Volltext gegenprüfen; falls nicht exakt belegt, als gerundete Spanne mit Quellbezug
formulieren. Nur wenn schnell machbar.

**Akzeptanzkriterium:** `bibtex` läuft ohne Warnungen über unzitierte/fehlende Keys;
jede Quelle in der Bibliografie ist im Text zitiert und deckt den behaupteten Inhalt.

---

## Arbeitspaket C — Länge auf ≤35 Seiten (Ziel ~32)

Aktuell 43 Seiten (Intro–Conclusion). Kürzen **nur durch Verdichtung von Prosa und
Entfernen von Redundanz** — keine Abbildungen/Tabellen/Resultate streichen. Stil gemäß
Riesen-Vorgaben + Academic Word List beibehalten.

**C1 — Time-Shift-Mechanismus entdoppeln (~−2 bis −3 S.).**
Die Erklärung „Autokorrelation → MSE belohnt Kopieren → Delay" steht 5–6×. Formal nur
**einmal** in `thesis/background.tex` 2.2.1 (MSE) herleiten. An den übrigen Stellen auf
je 1 Satz + Verweis kürzen:
- `thesis/introduction.tex` Abs. 3 (Kurzfassung behalten, Formaldetails raus)
- `thesis/experiments.tex` 4.1 (Baseline): nicht erneut herleiten, nur auf 2.2.1 verweisen
- `thesis/conclusions.tex` Synthese: Mechanismus nicht nochmals ausführen

**C2 — Kapitel 4 straffen (~−3 bis −4 S.).**
- `thesis/experiments.tex` 4.5 „Overall Comparison": von 4 `\paragraph`-Blöcken auf
  **einen** kurzen Absatz + Verweis kürzen (die Synthese gehört in Kap. 5).
- Pro Objective (4.2/4.3/4.4) je einen redundanten Interpretationsabsatz entfernen;
  Beobachtung + Interpretation zusammenfassen.

**C3 — Kapitel 2 verdichten (~−2 S.).**
- `thesis/background.tex`: die sehr langen Sub-Captions der Loss-Geometrie-Abbildung
  (Fig. mit `loss_mse`/`loss_bounded_lag`/`loss_dtw_alignment`) um ~50 % kürzen.
- Soft-DTW-Abschnitt (2.2.2) um einen Absatz kürzen; LSTM-Gating-Prosa straffen
  (Standardstoff knapper).

**C4 — Conclusion kürzen (~−1 S.).**
`thesis/conclusions.tex`: Antworten auf RQ1–RQ3 auf je 3–4 Sätze; Überschneidung mit
4.5 entfernen. Conclusion soll knapper sein als das Resultatkapitel.

**C5 — Related Work nicht doppeln.** Prüfen, ob die Related-Work-Absätze in der Einleitung
und in Kap. 2 inhaltlich überlappen; jeweils nur an **einer** Stelle ausführen.

**Akzeptanzkriterium:** Nach `pdflatex`+`bibtex`+2×`pdflatex` zeigt `main.toc`/PDF, dass
Intro–Conclusion ≤35 Seiten umfasst (Ziel ~32). Kein Resultat und keine Abbildung verloren.

---

## Arbeitspaket D — Anhang vollständig referenzieren

Jede Anhang-Komponente muss im Haupttext per `\ref` referenziert sein.
- Fig. A.1 `confusion_counts` → bereits referenziert (4.4.2). ✓
- Tab. A.1/A.2 (τ-Sweep) → bereits referenziert (4.4.1). ✓
- **Fig. A.2 `pr_paradox` (Precision–Recall-Raum) → aktuell NICHT referenziert.**

**D1 —** In `thesis/experiments.tex` an passender Stelle (Ende 4.4.2, wo der
Precision-Unterschied LSTM vs. PatchTST bzw. forecast-derived vs. direkt diskutiert wird)
einen Satz mit `Fig.~\ref{fig:app:pr_paradox}` ergänzen, z. B.: „The contrast between
forecast-derived and direct detectors in precision–recall space is shown in
Fig.~\ref{fig:app:pr_paradox}." — Falls die Abbildung inhaltlich nichts Neues beiträgt,
alternativ aus `thesis/appendix.tex` entfernen.

**Akzeptanzkriterium:** Keine Abbildung/Tabelle im Anhang ohne `\ref` im Haupttext.

---

## Arbeitspaket E — Eigenständigkeitserklärung

Die unterschriebene „Erklärung" (Art. 30 RSL Phil.-nat.) muss am **Ende** der Arbeit
stehen (Vorgabe Back Matter).

**E1 —** Neue Datei `thesis/declaration.tex` anlegen mit dem Standard-Wortlaut der
Universität Bern (philnat). Felder: Name/Vorname, Matrikelnummer, Studiengang
(Bachelor ankreuzen), Titel der Arbeit, Leiter der Arbeit (PD Dr. K. Riesen), Ort/Datum,
Unterschrift. Erklärungstext (selbstständig verfasst, nur angegebene Quellen, Plagiats-
Hinweis gemäß Art. 36 Abs. 1 UniG).

**E2 —** In `thesis/main.tex` ganz am Schluss (nach der Bibliografie) einbinden:
`\cleardoublepage` + `\include{declaration}`.

**E3 —** Platzhalter `\thesisauthororigin` = „[City of birth, Country]" in
`thesis/title.tex` entfernen (ungenutzt; Titelseite zeigt bereits „from Bern,
Switzerland").

**Akzeptanzkriterium:** PDF enthält als letzte Seite die Erklärung mit Unterschriftszeile;
Felder mit Roberts Daten ausgefüllt (Unterschrift erfolgt auf dem gedruckten Exemplar).

---

## Arbeitspaket F — Acknowledgements

Aktuell steht in `thesis/acknowledgments.tex` nur eine `planned`-Platzhalterbox.

**F1 —** Box durch echten Fliesstext ersetzen — **sachlich, ehrlich, nicht übertrieben**.
Vorschlag (anpassbar):
> I thank PD Dr. Kaspar Riesen for proposing the topic, for his supervision, and for his
> feedback during our discussions. I also thank the Pattern Recognition Group at the
> University of Bern for providing the data and computational resources used in this work.
(Nur erwähnen, was zutrifft — z. B. PRG-Daten/Ressourcen nur, wenn korrekt.)

**F2 —** Alle übrigen `planned`-Boxen im Projekt vor Abgabe entfernen (Suche nach
`\begin{planned}` in `thesis/`); die `planned`-Umgebung selbst kann in `main.tex` bleiben.

**Akzeptanzkriterium:** Keine `planned`-Box mehr im kompilierten PDF; Danksagung in
nüchternem Fliesstext.

---

## Reihenfolge & Abschluss

1. **A** (Datensatz) und **B** (Referenzen) — inhaltlich kritisch, zuerst.
2. **E** und **F** (Erklärung, Danksagung) — schnell erledigt, blocken die Abgabe sonst.
3. **D** (Anhang-Referenz) — kleiner Einzeleingriff.
4. **C** (Länge) — zuletzt, da es Text in allen Kapiteln berührt und nach den
   inhaltlichen Korrekturen erfolgen sollte.

**Nach allen Änderungen:** `pdflatex main` → `bibtex main` → `pdflatex main` ×2 ausführen
und prüfen: (i) ≤35 Seiten Intro–Conclusion, (ii) keine LaTeX-/BibTeX-Warnungen über
undefinierte Referenzen oder unzitierte Keys, (iii) Erklärung als letzte Seite vorhanden.
