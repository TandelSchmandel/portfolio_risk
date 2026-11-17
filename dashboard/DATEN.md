# Dashboard-Daten

Zwei Quellen:

* **Rechendienst und Website (Schema 2)**: Antworten von `src/portfolio_risk/api.py` und die Datei
  `risiko-dashboard-daten.js` der Website. Davon lebt die Dashboard-Seite.
* **Berichtsexport (Schema v1)**: `dashboard/data/*.json` aus den Ergebnissen von Schritt 2, für spätere
  Ansichten (Backtest, Stresstests, Modellvergleich).

Konventionen überall: VaR und ES sind **positive Verluste als Anteil** (0.055 = 5,5 %), Simulation ohne
Drift. Renditen, Drawdowns und Stresstest-Werte sind **negativ bei Verlust**. Datumsangaben im
ISO-Format. Portfolio-IDs: `A_Hauptportfolio`, `B_gleichgewichtet`, `C_chip_lastig`.

## Rechendienst (Schema 2)

Alle Endpunkte sind `GET`, Beschreibung und Ausprobieren unter `/docs`. Gewichte werden als
`TICKER:ZAHL` mit Komma übergeben (`NVDA:50,MSFT:30,AMZN:20`), die Skala ist frei, fehlende Ticker
zählen als 0. Ungültige Eingaben ergeben 422, zu viele Anfragen 429.

| Endpunkt | Inhalt |
|---|---|
| `/api/health` | `status`, `data_end`, `warm` (Szenarien und Snapshot vorgerechnet) |
| `/api/meta` | wie `meta` im Snapshot |
| `/api/snapshot` | Startdaten der Seite, siehe unten. Einmal je Start berechnet und gepackt ausgeliefert (etwa 60 KB) |
| `/api/curve?weights=…&horizon=10` | VaR/ES für eigene Gewichte über alle Konfidenzstufen, siehe unten |
| `/api/portfolio?weights=…&confidence=0.99&horizon=10&models=all` | Gewichte, Kennzahlen, Zeitreihen und VaR/ES je Modell (`series=false` lässt die Zeitreihen weg) |
| `/api/overview` | die drei vorgegebenen Portfolios in der Form von `/api/portfolio` |

### Snapshot (`/api/snapshot`, `window.RISK_SNAPSHOT`)

```
schema      2
generated   Zeitpunkt der Berechnung (UTC)
meta        data_start, data_end, points (Kurstage), trading_days (Renditen), trading_days_per_year (252),
            tickers [{symbol, name, group}], presets [{id, key (A/B/C), weights {Ticker: Anteil}}],
            models, main_model, student_t_nu, n_sims, ci_level (0.95), confidence_steps,
            limits {confidence [min, max], horizon [min, max]}, defaults {confidence, horizon, preset}
dates       [ISO-Datum] je Kurstag, der erste ist der Starttag
growth      {Ticker: [Kurs / Kurs am Starttag]}, so lang wie dates, 5 Nachkommastellen
cov         10 × 10, tägliche Kovarianz der Log-Renditen (Reihenfolge wie meta.tickers)
grid        model ("student_t"), ci_level, confidence [0.9 … 0.995 in 0.005], horizon [1 … 20],
            risk.<Portfolio>.{var, var_low, var_high, es}[Index Horizont][Index Konfidenz]
```

Daraus rechnet die Seite für beliebige Gewichte `w` (Summe 1): Wert `100 · Σ w_i · growth_i[t]`,
Drawdown `Wert / bisheriges Maximum − 1`, Rendite p. a. `(Endwert/100)^(252/trading_days) − 1`,
Volatilität p. a. `sqrt(w' cov w · 252)`. Das sind dieselben Formeln wie in `engine.series` und
`engine.kpis`.

### Kurve (`/api/curve`)

```
weights     {Ticker: Anteil}, normiert
horizon     Handelstage
data_end    letzter Kurstag des Dienstes (die Seite lädt die Startdaten neu, wenn er neuer ist)
model       student_t (wählbar mit model=)
ci_level    0.95
confidence  wie grid.confidence
var, var_low, var_high, es   je Konfidenzstufe
n           Zahl der Szenarien
```

## Berichtsexport (Schema v1)

Erzeugt mit `python scripts/export_dashboard.py` (etwa 35 s, mit `--ohne-backtest` wenige Sekunden).
Das Skript rechnet nichts neu simuliert, sondern liest `reports/v2/*.csv` und baut aus `build_dataset`
die Zeitreihen. Nach jedem neuen Lauf von `risk_report_v2.py` den Export wiederholen.

Fehlende Werte sind `null`.

### `data/meta.json` (2 KB)

| Feld | Inhalt |
|---|---|
| `schema_version` | 1 |
| `erstellt` | Zeitpunkt des Exports (UTC) |
| `datenbasis` | erster/letzter Renditetag, Anzahl Tagesrenditen, Student-t-ν, mittlere Korrelation |
| `ticker` | 10 Ticker in fester Reihenfolge |
| `portfolios[]` | `id`, `kurz` (A/B/C), `name`, `beschreibung`, `gewichte` {Ticker: Anteil} |
| `modelle` | Modell-ID → Klartext; `hauptmodell` = `student_t` |
| `konventionen` | Vorzeichen und Drift in Worten |

### `data/kennzahlen.json` (32 KB)

| Feld | Inhalt |
|---|---|
| `zusammenfassung` | {Portfolio: {Kennzahl: Wert}} wie `zusammenfassung.csv` |
| `var[]` | `portfolio`, `horizont` (1/10), `konfidenz` (0.95/0.99), `modell`, `modell_name`, `var`, `var_ki` [unten, oben], `es`, `n`, `nur_vergleich` |
| `backtest[]` | `portfolio`, `konfidenz`, `modell`, `tage`, `ausnahmen`, `erwartet`, `quote`, `kupiec_p`, `christoffersen_p`, `bedingte_abdeckung_p`, `mittlerer_var`, `ausnahmen_250`, `ampel` (nur 99 %) |
| `stresstests[]` | `gruppe` (Historisch/Datengetrieben/Hypothetisch), `szenario`, `detail`, `werte` {Portfolio: Wert} |
| `datenqualitaet[]` | je Ticker: Handelstage, fehlende Werte, größter Gewinn/Verlust (log), Ausreißer-Tage, Split-Verdacht |
| `titel` | {Ticker: `vola_pa`, `rendite_pa_log`} |
| `korrelation` | `ticker`, `matrix` (10 × 10, Log-Renditen) |
| `portfolio_vola_pa` | {Portfolio: annualisierte Volatilität aus `sqrt(w' Cov w)`} |

Modell-IDs: `historisch`, `normal`, `student_t`, `bootstrap`, `fhs`, `ewma_normal` (nur Backtest),
`historisch_wurzel10` (nur Vergleichszeile beim 10-Tages-VaR).

### `data/zeitreihen.json` (314 KB, gzip etwa 110 KB)

```
datum: [ISO-Datum] × 1 253
portfolios.<id>:
  rendite        tägliche Portfolio-Rendite (feste Gewichte)
  wert           Buy-and-hold-Wert, Start 100 am Tag vor dem ersten Renditetag
  drawdown       Rückgang vom bisherigen Höchststand (Buy-and-hold, wie stress.max_drawdown)
  vola_ewma_pa   EWMA-Volatilität (λ = 0,94), annualisiert, ohne Vorgriff
  backtest_99:   (fehlt mit --ohne-backtest)
    start        erster Testtag (Index in datum = 250)
    verlust      echter Verlust je Testtag (positiv = Verlust)
    var.<modell> am Vortag prognostizierter 99 %-VaR je Modell
```

Alle Arrays unter `portfolios.<id>` außer `backtest_99` sind genauso lang wie `datum`; die Arrays in
`backtest_99` beginnen bei `start` (1 003 Testtage).
