# portfolio-risk

Risikoanalyse für Portfolios auf Basis echter Börsendaten: historischer und
Monte-Carlo-Value-at-Risk, Stresstests und ein Dashboard.

## Status

- [x] Schritt 1: Datenladen und Renditeberechnung
- [ ] Schritt 2: VaR, Monte-Carlo, Stresstests
- [ ] Schritt 3: Dashboard
- [ ] Schritt 4: Demo und Dokumentation

## Daten

Kurse als CSV-Datei pro Wertpapier in `data/raw/` ablegen. Jede Datei braucht die
Spalten `Date` und `Adj Close`, der Dateiname ohne Endung ist der Ticker (`AAPL.csv`).
Das Format entspricht dem Download-Export von Yahoo Finance.

## Entwicklung

```bash
pip install -r requirements.txt
pytest
```
