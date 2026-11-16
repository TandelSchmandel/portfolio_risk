# Daten, Renditen und Kovarianzmatrix: Annahmen

Modul: `src/portfolio_risk/dataset.py` (`build_dataset`, `RiskDataset`), Tests: `tests/test_dataset.py`.

```python
from portfolio_risk.dataset import build_dataset
from portfolio_risk.portfolios import TICKERS

ds = build_dataset("data/raw", TICKERS)
ds.log_returns, ds.simple_returns, ds.cov, ds.corr, ds.vol_annual, ds.summary()
```

## Was berechnet wird

* **Kurse:** `Adj Close` (um Splits und Dividenden bereinigt). Nur Handelstage, an denen alle 10 Titel einen Kurs haben,
  damit alle Renditen und Kovarianzen über dieselben Tage laufen. Ergebnis: 17.11.2021 bis 13.11.2026, 1 253 Tagesrenditen.
* **Log-Rendite:** `ln(P_t / P_{t-1})`, additiv über die Zeit. **Einfache Rendite:** `P_t / P_{t-1} − 1`, additiv über Titel
  (für Portfolio-Renditen mit festen Gewichten).
* **Kovarianzmatrix:** Stichprobenkovarianz (Nenner n − 1) der **Log-Renditen**, pro Handelstag.
  Annualisiert mit 252 Handelstagen (Kovarianz × 252, Volatilität × √252).

## Annahmen

1. Renditen sind über die Zeit stationär; Kovarianz und Korrelation sind für die 5 Jahre konstant. Das stimmt nur grob:
   Korrelationen steigen in Krisen. Dafür gibt es in Schritt 2 gestresste Korrelationen und die EWMA-/FHS-Modelle.
2. Fehlende Tage werden nicht aufgefüllt, sondern entfernt. In den vorliegenden Daten fehlt nichts.
3. Kurse müssen positiv sein, sonst bricht `build_dataset` ab. Die Datenprüfung (`quality.py`) meldet Ausreißer und Split-Fehler.
4. Keine Drift-Bereinigung in der Kovarianz (Mittelwert wird abgezogen, aber nicht in die Matrix eingerechnet).
   Der Mittelwert 2021 bis 2026 ist mit rund 26 % p. a. (Durchschnitt der Titel) außergewöhnlich hoch und sollte nicht fortgeschrieben werden.
5. Die Portfolio-Volatilität `sqrt(w' Cov w)` ist eine Näherung: sie rechnet mit Log-Renditen und festen Gewichten.

## Kennzahlen der echten Daten

| | |
|---|---|
| Positiv definit | ja |
| Kleinster Eigenwert | 1,4 · 10⁻⁴ (täglich) |
| Konditionszahl | 35 (gut invertierbar) |
| Mittlere paarweise Korrelation | 0,48 |
| Volatilität p. a. | MSFT 28 %, GOOGL 33 %, AMZN 37 %, TSM 39 %, AVGO 45 %, META 46 %, NVDA 52 %, MU 58 %, AMD 59 %, PLTR 66 % |
| Tägliche Portfolio-Volatilität | A 2,10 %, B 2,13 %, C 2,34 % |

Die Chip-Titel korrelieren untereinander mit 0,55 bis 0,69, die Plattformen untereinander mit 0,38 bis 0,62.

## Grenzen

* Mit 10 Titeln und 1 253 Tagen ist die Matrix gut bestimmt. `shrunk_covariance` (Ledoit-Wolf, Schrumpfungsintensität 1,0 %)
  weicht um unter 0,5 % von der Stichprobenmatrix ab. Das ist eine Gegenprobe, kein Ersatz.
* Fünf Jahre enthalten nur ein Marktregime mit Zinswende und KI-Boom; Kovarianzen in anderen Regimen (2008, 2020) fehlen.
* Die Schätzung ist nicht robust gegen einzelne Extremtage (META 03.02.2022 −26 %, PLTR 06.02.2024 +31 %). Sie bleiben bewusst drin.
