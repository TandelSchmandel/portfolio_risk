# Schritt 2, zweiter Durchlauf: Methodik, Ergebnisse und Grenzen

Erzeugt mit `python scripts/risk_report_v2.py` (Laufzeit etwa 75 s, fester Seed `20261116`).
Der erste Durchlauf (`scripts/risk_report.py`, `reports/risk_summary.csv`) bleibt unverändert als Vergleich.

| Datei | Inhalt |
|---|---|
| `datenqualitaet.csv` | Prüfung jeder Kursdatei |
| `var_modelle.csv` | VaR und ES je Portfolio, Modell, Konfidenz (95/99 %) und Horizont (1/10 Tage), mit 95 %-Konfidenzintervall des VaR |
| `backtest.csv` | Ausnahmen, Kupiec-, Christoffersen- und Ampeltest je Modell |
| `stresstests.csv` | 21 Szenarien in drei Gruppen |
| `zusammenfassung.csv` | Kernzahlen je Portfolio |

Konvention: VaR und ES sind positive Verluste in Prozent des Portfoliowerts (0.055 = 5,5 %).
Stresstest-Renditen sind negativ bei Verlust.

## Was sich gegenüber dem ersten Durchlauf ändert

| Schwäche in v1 | Lösung in v2 |
|---|---|
| Monte-Carlo nur mit Normalverteilung, Fat Tails unterschätzt | Vier Simulationsmodelle: Normal (Vergleich), multivariate **Student-t** mit per Maximum-Likelihood geschätzten Freiheitsgraden, **Block-Bootstrap** echter Tagesvektoren, **gefilterte historische Simulation** (FHS) mit aktueller EWMA-Volatilität |
| 100 000 Simulationen ohne Fehlerangabe | **1 000 000** Simulationen je Modell mit festem Seed; **95 %-Konfidenzintervall** des VaR aus Ordnungsstatistiken, auch für den historischen VaR |
| Kein Backtest | Rollierendes Out-of-sample-**Backtesting** (250-Tage-Fenster, 1 003 Testtage), Ausnahmen zählen, **Kupiec**-Test, **Christoffersen**-Unabhängigkeitstest, bedingte Abdeckung, **Basler Ampel** |
| Nur Tages-VaR | **10-Tages-VaR** durch Simulation ganzer 10-Tages-Pfade bzw. echte überlappende 10-Tages-Fenster; Wurzel-Zeit nur noch als Vergleichszeile |
| Einfache Renditen, keine Datenprüfung | **Log-Renditen** aus `Adj Close` für Zeitaggregation, einfache Renditen für die Gewichtung über Titel; **Datenprüfung** auf Lücken, Ausreißer (robuster z-Wert), Split-Fehler |
| Drei Stresstests | **21 Szenarien** in drei klar getrennten Gruppen (historisch, datengetrieben, hypothetisch) |

## Datengrundlage und Datenprüfung

* 10 Titel, `Adj Close` (um Splits und Dividenden bereinigt), 17.11.2021 bis 13.11.2026, 1 253 Tagesrenditen,
  nur Tage, an denen alle Titel gehandelt wurden.
* Keine fehlenden Werte, keine doppelten Daten, keine Lücken über 5 Kalendertage, keine Split-Fehler
  (die Splits von NVDA 2024 und AVGO 2024 sind sauber bereinigt; das Verhältnis Adj Close/Close springt nirgends um mehr als 1 %).
* 11 Tage mit robustem z-Wert über 8, z. B. META 03.02.2022 (−26 %), AVGO 13.12.2024 (+24 %), PLTR 06.02.2024 (+31 %).
  Alle fallen auf Quartalszahlen oder bekannte Marktereignisse. Sie sind **echte Bewegungen und bleiben drin**,
  denn genau diese Tage bestimmen den 99 %-VaR. Entfernen würde das Risiko schönrechnen.

### Eine Datenquelle für alle Rechnungen

Seit dem Datenmodul (`build_dataset`, Annahmen in [`docs/ANNAHMEN_DATEN.md`](../../docs/ANNAHMEN_DATEN.md)) kommen alle
Eingangsgrößen aus einem `RiskDataset`: die Log-Renditen für Simulation, Stresstests und historische Fenster, die einfachen
Renditen für die täglichen Portfolio-Renditen im Backtest und **eine** Kovarianzmatrix `ds.cov` für Normal-MC, Student-t-MC
(inklusive Schätzung von ν), korrelierte Schocks und den Parameterstress. Bootstrap und FHS ziehen direkt aus `ds.log_returns`.

`reports/v2/datenbasis.csv` dokumentiert die Kennzahlen des Datenmoduls und prüft die Konsistenz: Die Tagesvolatilität
`sqrt(w' Cov w)` (A 2,100 %, B 2,127 %, C 2,339 %) stimmt mit der Streuung der Normal-Simulation bis auf 0,1 % überein.
Weil die Rohdaten und die Stichprobenkovarianz dieselben sind wie im vorherigen Lauf, sind alle Ergebnisse unverändert
(Abweichung 0 in `var_modelle.csv`, `backtest.csv` und `stresstests.csv`).

## Rendite-Handling

* Log-Rendite je Titel: `r = ln(P_t / P_{t-1})`. Über die Zeit additiv, daher werden Mehrtagespfade durch Summieren gebildet.
* Portfolio über einen Horizont (Buy-and-hold, keine Umschichtung): `R_p = Σ w_i · (exp(Σ_t r_it) − 1)`.
  Einfache Renditen sind über Titel additiv, Log-Renditen nicht.
* Simulationen laufen **ohne Drift**. Die Durchschnittsrendite 2021 bis 2026 war für KI-Titel außergewöhnlich hoch;
  sie fortzuschreiben, würde den 10-Tages-VaR um ein bis zwei Prozentpunkte verkleinern. Der rein historische VaR
  enthält die Drift der Stichprobe zwangsläufig.

## Ergebnisse (Portfolio A, Hauptportfolio)

| Kennzahl | 1 Tag | 10 Tage |
|---|---|---|
| VaR 99 % historisch [95 %-KI] | 5,51 % [4,92 – 6,39] | 12,92 % [9,89 – 19,16] |
| VaR 99 % Normal (Methode v1) | 4,75 % | 14,06 % |
| VaR 99 % Student-t (ν = 4,75) | 5,28 % | 14,45 % |
| VaR 99 % Block-Bootstrap | 5,62 % | 14,21 % |
| VaR 99 % FHS (heutige Volatilität) | 6,69 % | 17,42 % |
| ES 99 % Student-t | 6,96 % | 16,98 % |
| Historischer 1-Tages-VaR × √10 | – | 17,42 % |

Alle drei Portfolios im Vergleich (`zusammenfassung.csv`):

| | A Haupt | B gleich | C Chips |
|---|---|---|---|
| VaR 99 % 1 Tag Student-t | 5,28 % | 5,35 % | 5,88 % |
| VaR 99 % 1 Tag FHS | 6,69 % | 7,06 % | 7,91 % |
| VaR 99 % 10 Tage Student-t | 14,45 % | 14,59 % | 15,97 % |
| ES 99 % 1 Tag Student-t | 6,96 % | 7,04 % | 7,73 % |
| Maximaler Drawdown | −49,9 % | −49,7 % | −49,9 % |
| Schlechtester Tag (27.01.2025) | −8,0 % | −8,7 % | −11,1 % |

**Lesart:**

1. **Die Normalverteilung unterschätzt den 99 %-VaR um rund 10 bis 18 %** gegenüber Student-t, Bootstrap und Historie
   (4,75 % statt 5,3 bis 5,6 %). Bei 95 % ist es umgekehrt: Fat-Tail-Verteilungen haben mehr Masse in der Mitte,
   daher liegt der Student-t-VaR 95 % sogar leicht unter dem normalen. Die Normalverteilung ist bei 95 % zu vorsichtig,
   bei 99 % zu optimistisch.
2. **Monte-Carlo-Fehler ist jetzt vernachlässigbar** (KI-Breite um 0,05 Prozentpunkte bei 10⁶ Läufen).
   Die eigentliche Unsicherheit liegt in den **Daten**: Der historische 99 %-VaR hat ein KI von 4,9 bis 6,4 %,
   beim 10-Tages-VaR sogar 9,9 bis 19,2 %, weil 5 Jahre nur etwa 125 unabhängige 10-Tages-Fenster liefern.
3. **Der FHS-VaR liegt deutlich über allen anderen**, weil die Volatilität Anfang November 2026 hoch ist
   (u. a. AMD und MU −17 % am 05.11.2026). FHS beantwortet die Frage „Wie riskant ist morgen?“,
   die anderen Modelle „Wie riskant war ein durchschnittlicher Tag 2021 bis 2026?“.
4. **Wurzel-Zeit ist hier zufällig nah am FHS-Wert, aber aus dem falschen Grund.** Gegenüber den
   unbedingten Modellen (12,9 bis 14,5 %) überschätzt √10 den 10-Tages-VaR, weil der Tages-VaR von Fat Tails
   getrieben ist, die sich über 10 Tage teilweise ausmitteln.
5. **Portfolio C hat durchgehend einen 10 bis 20 % höheren VaR** als A und B (beim schlechtesten Tag sogar fast 40 %). A und B unterscheiden sich kaum,
   weil beide 50 % Chips halten und die Titel hoch korreliert sind.

## Backtest (1 003 Tage, 250-Tage-Fenster, je Tag neu geschätzt)

99 %-VaR, Ausnahmen (erwartet: 10):

| Modell | A | B | C | Bewertung |
|---|---|---|---|---|
| Historisch | 11 | 12 | 14 | besteht Kupiec |
| Normal | 13 | 16 | 14 | besteht Kupiec, aber B und C gelbe Ampel |
| **Student-t** | **11** | **11** | **11** | **bestes Modell: Kupiec p = 0,76, alle Ampeln grün** |
| EWMA-Normal | 21 | 22 | 18 | **verworfen** (Kupiec p < 0,03) |
| FHS (EWMA) | 13 | 14 | 12 | besteht, beste Unabhängigkeit (Christoffersen p ≈ 0,55) |

Bei 95 % wird die Normalverteilung für alle drei Portfolios verworfen (zu **wenige** Ausnahmen, Kupiec p < 0,05),
Student-t und FHS bestehen. Kein Modell zeigt signifikant gehäufte Ausnahmen (alle Christoffersen p > 0,05).

**Empfehlung:** Student-t als Hauptmodell für den VaR, FHS als tagesaktuelle Ergänzung, ES 99 % als Maß für den Verlust
jenseits des VaR. Die Normalverteilung nur noch als Vergleich.

## Stresstests

**Gruppe 1, historische Szenarien** (feste Daten, Buy-and-hold): Zinswende 2022 (−47 % für A), Ausverkauf Q2 2022 (−30 %),
Sommer 2024 (−20 %), Zollschock April 2025 (−13 %), DeepSeek-Tag (−8 %), META-Gewinnwarnung (−7 %).

**Gruppe 2, datengetrieben** (keine Auswahl nötig): schlechteste Fenster über 1/5/20/60 Tage (−8 / −14 / −22 / −30 % für A),
maximaler Drawdown −50 % (Hoch 19.11.2021, Tief 03.11.2022, erholt 14.06.2023), Stressed VaR 99 % nach Basel 2.5: 6,2 %.

**Gruppe 3, hypothetisch:**
* Pauschal: alle Titel −20 %; Chips −30 % und Plattformen −15 %; „KI-Blase platzt“ Chips −50 % und Plattformen −30 % (A: −40 %, C: −45 %).
* Korreliert: Ein Titel oder Korb fällt, die übrigen bewegen sich um ihren bedingten Erwartungswert unter der historischen
  Kovarianz. NVDA −30 % zieht das Portfolio A im Mittel um 18 % nach unten, der Chip-Korb −30 % um 23 %.
* Parameterstress: Korrelationen auf 0,9 erhöhen den Student-t-VaR 99 % von 5,4 auf 6,8 %, Volatilität ×1,5 auf 7,9 %.

## Annahmen

* Gewichte konstant; Tages-VaR mit täglichem Rebalancing, Mehrtages-VaR und Stresstests als Buy-and-hold.
* Kein Drift in den Simulationen (siehe oben). Keine Transaktionskosten, Steuern oder Währungseffekte (alles in USD).
* Student-t: ein gemeinsames ν für alle Titel; Kovarianz gleich der Stichprobenkovarianz.
* FHS und EWMA: λ = 0,94 (RiskMetrics); Volatilität pro Titel gefiltert, Korrelation über ganze historische Residuenvektoren.
* Block-Bootstrap: Blocklänge 5 Tage beim 10-Tages-Horizont, zirkulär.
* Backtest auf Ebene der Portfolio-Tagesrendite. Für Student-t ist das exakt dasselbe Modell wie die multivariate
  Simulation (Linearkombination einer multivariaten t ist t mit gleichem ν).

## Grenzen

* **Nur fünf Jahre Daten (ab November 2021).** Die Finanzkrise 2008, der Dotcom-Crash 2000 und der Corona-Crash 2020
  sind nicht enthalten. Ein echtes Platzen einer KI-Blase kann deutlich schlimmer sein als alles in den Daten;
  dafür gibt es die hypothetischen Szenarien in Gruppe 3, deren Höhe aber gesetzt und nicht geschätzt ist.
* **Die Daten beginnen fast am Hoch von 2021.** Der maximale Drawdown und der Stressed VaR beginnen am ersten Datentag;
  ein Teil des vorangehenden Anstiegs und des Rückgangs liegt außerhalb der Stichprobe.
* **Survivorship Bias:** Ausgewählt wurden Titel, die heute groß sind. Die historische Rendite ist dadurch zu gut.
* **PLTR** ist erst seit 2020 börsennotiert und hatte die höchste Volatilität; seine Schätzungen sind am unsichersten.
* Das Konfidenzintervall des VaR deckt Stichproben- bzw. Simulationsfehler ab, **nicht den Modellfehler**.
  Wie weit die Modelle auseinanderliegen (4,75 bis 6,69 % beim 1-Tages-VaR 99 %), ist das ehrlichere Maß für die Unsicherheit.
* Beim Bootstrap ist das Monte-Carlo-KI für einen Tag fast null breit, weil nur 1 253 verschiedene Tage gezogen werden können;
  die maßgebliche Unsicherheit steht beim historischen VaR.
* Überlappende 10-Tages-Fenster sind abhängig; das KI des historischen 10-Tages-VaR rechnet deshalb mit n/10 effektiven Beobachtungen.
* Backtest mit 1 003 Tagen hat wenig Trennschärfe bei 99 % (erwartet nur 10 Ausnahmen); Unterschiede von 2 bis 3 Ausnahmen sind nicht signifikant.
* Liquidität: Es wird angenommen, dass sich Positionen zum Schlusskurs auflösen lassen. Bei diesen Titeln realistisch, in einem echten Crash mit Kurslücken über Nacht nicht garantiert.
