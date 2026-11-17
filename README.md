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
- [x] Schritt 3: Dashboard als Unterseite der eigenen Website (Repository `Elias-Tandel.github.io`)
  - Rechendienst (FastAPI): `src/portfolio_risk/api.py` auf dem Rechenkern `engine.py`, VaR und ES für
    frei wählbare Gewichte, Konfidenz und Horizont
  - tägliche Aktualisierung der Kurse per GitHub Actions: `scripts/update_prices.py`,
    `.github/workflows/update-data.yml`
  - Aufbau: [`dashboard/ARCHITEKTUR.md`](dashboard/ARCHITEKTUR.md), Datenformat:
    [`dashboard/DATEN.md`](dashboard/DATEN.md), Einrichten und Betrieb:
    [`dashboard/BETRIEB.md`](dashboard/BETRIEB.md)
  - Export der Berichtsergebnisse für spätere Ansichten: `scripts/export_dashboard.py` → `dashboard/data/*.json`
- [ ] Schritt 4: Demo und Dokumentation

## Daten

Kurse als CSV-Datei pro Wertpapier in `data/raw/` ablegen. Jede Datei braucht die
Spalten `Date` und `Adj Close`, der Dateiname ohne Endung ist der Ticker (`AAPL.csv`).
Das Format entspricht dem Download-Export von Yahoo Finance. `scripts/update_prices.py` schreibt die
Dateien fort (lädt, prüft, ersetzt nur bei bestandener Prüfung).

## Entwicklung

```bash
pip install -r requirements.txt -r requirements-api.txt
pytest
python scripts/risk_report_v2.py   # etwa 75 Sekunden
python scripts/export_dashboard.py # Berichtsergebnisse als JSON, etwa 35 Sekunden
uvicorn portfolio_risk.api:create_app --factory --app-dir src --port 8000   # Rechendienst, Doku unter /docs
python scripts/build_snapshot.py --out ../Elias-Tandel.github.io/risiko-dashboard-daten.js
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
| `engine.py` | Rechenkern des Dashboards: Szenarien je Titel im Speicher, Kennzahlen für beliebige Gewichte in Millisekunden |
| `api.py` | HTTP-Schnittstelle (FastAPI) für die Dashboard-Seite |
| `update.py` | Kurse von Yahoo Finance laden, gegen den alten Stand prüfen, Dateien ersetzen |
| `portfolios.py` | Ticker, Namen, Gruppen und die drei vorgegebenen Portfolios |
