# portfolio-risk

Risikoanalyse für Portfolios auf Basis echter Börsendaten: historischer und
Monte-Carlo-Value-at-Risk, Stresstests und ein Dashboard.

## Status

- [x] Schritt 1: Datenladen und Renditeberechnung
- [x] Schritt 2: VaR, Monte-Carlo, Stresstests
  - erster Durchlauf: `scripts/risk_report.py` → `reports/risk_summary.csv`
  - zweiter Durchlauf mit höherer Genauigkeit (Student-t, Bootstrap, FHS, Backtesting, 10-Tages-VaR,
    21 Stresstests): `scripts/risk_report_v2.py` → `reports/v2/`, Methodik und Grenzen in
    [`reports/v2/METHODIK.md`](reports/v2/METHODIK.md)
  - alle Eingangsgrößen (Renditen, Kovarianzmatrix) kommen aus `build_dataset`; Konsistenzprüfung in
    `reports/v2/datenbasis.csv`
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
python scripts/risk_report_v2.py   # etwa 75 Sekunden
```

## Module

| Modul | Inhalt |
|---|---|
| `data.py` | Kurse laden, Renditen |
| `dataset.py` | `build_dataset`: Kurse → Log- und einfache Renditen → Kovarianzmatrix (Annahmen: [`docs/ANNAHMEN_DATEN.md`](docs/ANNAHMEN_DATEN.md)) |
| `quality.py` | Datenprüfung: Lücken, Ausreißer, Split-Fehler |
| `models.py` | Simulation über h Tage: Normal, Student-t, Block-Bootstrap, gefilterte historische Simulation |
| `risk.py` | VaR, ES, Konfidenzintervall des VaR |
| `backtest.py` | Rollierende VaR-Prognosen, Kupiec, Christoffersen, Basler Ampel |
| `stress.py` | Historische, datengetriebene und hypothetische Stresstests |
