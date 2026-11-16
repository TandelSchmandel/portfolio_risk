# Dashboard-Daten (Schema v1)

Erzeugt mit `python scripts/export_dashboard.py` (etwa 35 s, mit `--ohne-backtest` wenige Sekunden).
Das Skript rechnet nichts neu simuliert, sondern liest `reports/v2/*.csv` und baut aus `build_dataset`
die Zeitreihen. Nach jedem neuen Lauf von `risk_report_v2.py` den Export wiederholen.

Konventionen: VaR und ES sind **positive Verluste als Anteil** (0.055 = 5,5 %). Renditen, Drawdowns und
Stresstest-Werte sind **negativ bei Verlust**. Datumsangaben im ISO-Format, fehlende Werte sind `null`.
Portfolio-IDs: `A_Hauptportfolio`, `B_gleichgewichtet`, `C_chip_lastig`.

## `data/meta.json` (2 KB)

| Feld | Inhalt |
|---|---|
| `schema_version` | 1 |
| `erstellt` | Zeitpunkt des Exports (UTC) |
| `datenbasis` | erster/letzter Renditetag, Anzahl Tagesrenditen, Student-t-ν, mittlere Korrelation |
| `ticker` | 10 Ticker in fester Reihenfolge |
| `portfolios[]` | `id`, `kurz` (A/B/C), `name`, `beschreibung`, `gewichte` {Ticker: Anteil} |
| `modelle` | Modell-ID → Klartext; `hauptmodell` = `student_t` |
| `konventionen` | Vorzeichen und Drift in Worten |

## `data/kennzahlen.json` (32 KB)

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

## `data/zeitreihen.json` (314 KB, gzip etwa 110 KB)

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
