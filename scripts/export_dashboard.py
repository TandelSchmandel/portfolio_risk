"""Exportiert die Ergebnisse aus Schritt 2 als JSON für das Dashboard (Schritt 3).

Liest die fertigen Tabellen aus reports/v2/ (keine neue Simulation) und berechnet aus
build_dataset die Zeitreihen, die ein Dashboard für Charts braucht. Ausgabe in
dashboard/data/:

    meta.json        Stand, Datenbasis, Portfolios mit Gewichten, Ticker, Methodik-Hinweise
    kennzahlen.json  VaR/ES je Modell, Backtest, Stresstests, Datenqualität, Korrelationen
    zeitreihen.json  Wertentwicklung, Drawdown, EWMA-Volatilität, Backtest-VaR je Portfolio

Konventionen (wie in reports/v2/METHODIK.md): VaR und ES sind positive Verluste als Anteil
(0.055 = 5,5 %), Renditen und Stresstest-Werte sind negativ bei Verlust. Datumsangaben im
ISO-Format. Schema in dashboard/DATEN.md.

    python scripts/export_dashboard.py              # etwa 35 Sekunden (Backtest-Prognosen)
    python scripts/export_dashboard.py --ohne-backtest   # wenige Sekunden
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from portfolio_risk.backtest import rolling_var_forecasts  # noqa: E402
from portfolio_risk.data import portfolio_returns  # noqa: E402
from portfolio_risk.dataset import TRADING_DAYS, build_dataset  # noqa: E402
from portfolio_risk.models import EWMA_LAMBDA  # noqa: E402
from portfolio_risk.portfolios import PORTFOLIOS, TICKERS  # noqa: E402

REPORTS = ROOT / "reports" / "v2"
OUT = ROOT / "dashboard" / "data"
SCHEMA_VERSION = 1
DIGITS = 6

PORTFOLIO_LABELS = {
    "A_Hauptportfolio": {"kurz": "A", "name": "Hauptportfolio", "beschreibung": "NVDA 18 %, übrige Titel 6 bis 12 %"},
    "B_gleichgewichtet": {"kurz": "B", "name": "Gleichgewichtet", "beschreibung": "Alle zehn Titel je 10 %"},
    "C_chip_lastig": {"kurz": "C", "name": "Chip-lastig", "beschreibung": "Halbleiter 75 %, Plattformen 25 %"},
}

MODEL_IDS = {
    "Historisch": "historisch",
    "Normal (MC)": "normal",
    "Student-t (MC)": "student_t",
    "Block-Bootstrap (MC)": "bootstrap",
    "FHS (EWMA, MC)": "fhs",
    "Normal": "normal",
    "Student-t": "student_t",
    "EWMA-Normal": "ewma_normal",
    "FHS (EWMA)": "fhs",
    "Historisch x Wurzel(10) (nur Vergleich)": "historisch_wurzel10",
}


def clean(x):
    """Macht Werte JSON-tauglich: NaN -> None, numpy -> Python, Floats gerundet."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else round(x, DIGITS)
    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()
    return x


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def model_id(name: str) -> str:
    return MODEL_IDS.get(name, name.lower().replace(" ", "_"))


# --------------------------------------------------------------------------- Kennzahlen aus reports/v2


def var_records() -> list[dict]:
    df = pd.read_csv(REPORTS / "var_modelle.csv")
    return [
        {
            "portfolio": r["Portfolio"],
            "horizont": int(r["Horizont (Tage)"]),
            "konfidenz": float(r["Konfidenz"]),
            "modell": model_id(r["Modell"]),
            "modell_name": r["Modell"],
            "var": r["VaR"],
            "var_ki": [r["VaR KI unten"], r["VaR KI oben"]],
            "es": r["ES"],
            "n": None if pd.isna(r["n"]) else int(r["n"]),
            "nur_vergleich": "nur Vergleich" in r["Modell"],
        }
        for _, r in df.iterrows()
    ]


def backtest_records() -> list[dict]:
    df = pd.read_csv(REPORTS / "backtest.csv")
    out = []
    for _, r in df.iterrows():
        out.append(
            {
                "portfolio": r["Portfolio"],
                "konfidenz": float(r["Konfidenz"]),
                "modell": model_id(r["Modell"]),
                "modell_name": r["Modell"],
                "tage": int(r["Tage"]),
                "ausnahmen": int(r["Ausnahmen"]),
                "erwartet": r["Erwartet"],
                "quote": r["Quote"],
                "kupiec_p": r["Kupiec p"],
                "christoffersen_p": r["Christoffersen Unabh. p"],
                "bedingte_abdeckung_p": r["Bedingte Abdeckung p"],
                "mittlerer_var": r["Mittlerer VaR"],
                "ausnahmen_250": None if pd.isna(r["Ausnahmen letzte 250 Tage"]) else int(r["Ausnahmen letzte 250 Tage"]),
                "ampel": None if pd.isna(r["Basler Ampel"]) else r["Basler Ampel"],
            }
        )
    return out


def stress_records() -> list[dict]:
    df = pd.read_csv(REPORTS / "stresstests.csv")
    out = []
    for (gruppe, szenario), g in df.groupby(["Gruppe", "Szenario"], sort=False):
        out.append(
            {
                "gruppe": gruppe.split(" ", 1)[1] if gruppe[0].isdigit() else gruppe,
                "szenario": szenario,
                "detail": g["Detail"].fillna("").iloc[0],
                "werte": dict(zip(g["Portfolio"], g["Wert"])),
            }
        )
    return out


def quality_records() -> list[dict]:
    df = pd.read_csv(REPORTS / "datenqualitaet.csv")
    return [
        {
            "ticker": r["Ticker"],
            "erster_tag": r["Erster Tag"],
            "letzter_tag": r["Letzter Tag"],
            "handelstage": int(r["Handelstage"]),
            "fehlende_werte": int(r["Fehlende Werte"]),
            "groesster_gewinn_log": r["Größter Tagesgewinn (log)"],
            "groesster_verlust_log": r["Größter Tagesverlust (log)"],
            "ausreisser": int(r["Ausreißer |robust z| > 8"]),
            "ausreisser_tage": [] if pd.isna(r["Ausreißer-Tage"]) else str(r["Ausreißer-Tage"]).split(";"),
            "split_verdacht": int(r["Split-Verdacht"]),
        }
        for _, r in df.iterrows()
    ]


def summary_table() -> dict:
    df = pd.read_csv(REPORTS / "zusammenfassung.csv", index_col=0)
    return {col: df[col].to_dict() for col in df.columns}


# --------------------------------------------------------------------------- Zeitreihen aus build_dataset


def ewma_vol(r: pd.Series, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """EWMA-Tagesvolatilität wie im Backtest: var[t] nutzt nur Daten bis t-1."""
    x = r.to_numpy(dtype=float)
    var = np.empty_like(x)
    var[0] = x[:30].var(ddof=1)
    for t in range(1, len(x)):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2
    return np.sqrt(var)


def portfolio_series(log_returns: pd.DataFrame, weights, r: pd.Series, with_backtest: bool) -> dict:
    """Zeitreihen eines Portfolios.

    Wert und Drawdown sind Buy-and-hold ab dem ersten Tag (wie der maximale Drawdown in
    stress.max_drawdown), Rendite, Volatilität und Backtest nutzen die täglichen Portfolio-
    Renditen mit festen Gewichten (wie backtest.py).
    """
    w = np.asarray(weights, dtype=float)
    wealth = pd.Series(np.exp(log_returns.cumsum()).to_numpy() @ w, index=log_returns.index)
    drawdown = wealth / np.maximum(wealth.cummax(), 1.0) - 1  # Startwert 1 zählt als erster Höchststand
    vol = ewma_vol(r)
    series = {
        "rendite": r.to_numpy(),
        "wert": wealth.to_numpy() * 100,  # Startwert 100 am Tag vor dem ersten Renditetag
        "drawdown": drawdown.to_numpy(),
        "vola_ewma_pa": vol * np.sqrt(TRADING_DAYS),
    }
    out = {k: [clean(v) for v in vals] for k, vals in series.items()}
    if with_backtest:
        f = rolling_var_forecasts(r, confidence=0.99)
        out["backtest_99"] = {
            "start": f.index[0].date().isoformat(),
            "verlust": [clean(v) for v in f["Verlust"]],
            "var": {model_id(m): [clean(v) for v in f[m]] for m in f.columns if m != "Verlust"},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ohne-backtest", action="store_true", help="rollierende VaR-Prognosen weglassen (schneller)")
    args = parser.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = build_dataset(ROOT / "data" / "raw", TICKERS)
    daily = {name: portfolio_returns(ds.simple_returns, w) for name, w in PORTFOLIOS.items()}
    dates = [d.date().isoformat() for d in ds.simple_returns.index]

    basis = pd.read_csv(REPORTS / "datenbasis.csv", index_col=0)["A_Hauptportfolio"]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "erstellt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datenbasis": {
            "erster_renditetag": dates[0],
            "letzter_renditetag": dates[-1],
            "tagesrenditen": len(dates),
            "kursfeld": "Adj Close",
            "student_t_nu": float(basis["Student-t nu"]),
            "mittlere_korrelation": float(basis["Mittlere Korrelation"]),
        },
        "ticker": TICKERS,
        "portfolios": [
            {"id": name, **PORTFOLIO_LABELS[name], "gewichte": dict(zip(TICKERS, w))} for name, w in PORTFOLIOS.items()
        ],
        "modelle": {
            "historisch": "Historische Simulation",
            "normal": "Monte-Carlo, Normalverteilung (Vergleich)",
            "student_t": "Monte-Carlo, multivariate Student-t (Hauptmodell)",
            "bootstrap": "Monte-Carlo, Block-Bootstrap",
            "fhs": "Gefilterte historische Simulation mit aktueller EWMA-Volatilität",
            "ewma_normal": "Normalverteilung mit EWMA-Volatilität (nur Backtest)",
        },
        "hauptmodell": "student_t",
        "konventionen": {
            "var_es": "positiver Verlust als Anteil des Portfoliowerts",
            "renditen_stress": "negativ bei Verlust",
            "drift": "Simulationen ohne Drift",
        },
    }

    kennzahlen = {
        "schema_version": SCHEMA_VERSION,
        "zusammenfassung": summary_table(),
        "var": var_records(),
        "backtest": backtest_records(),
        "stresstests": stress_records(),
        "datenqualitaet": quality_records(),
        "titel": {
            t: {"vola_pa": ds.vol_annual[t], "rendite_pa_log": ds.mean_daily[t] * TRADING_DAYS} for t in TICKERS
        },
        "korrelation": {"ticker": TICKERS, "matrix": ds.corr.loc[TICKERS, TICKERS].to_numpy().tolist()},
        "portfolio_vola_pa": {name: ds.portfolio_volatility(w) * np.sqrt(TRADING_DAYS) for name, w in PORTFOLIOS.items()},
    }

    zeitreihen = {
        "schema_version": SCHEMA_VERSION,
        "datum": dates,
        "portfolios": {
            name: portfolio_series(ds.log_returns, PORTFOLIOS[name], r, not args.ohne_backtest) for name, r in daily.items()
        },
    }

    write_json(OUT / "meta.json", meta)
    write_json(OUT / "kennzahlen.json", kennzahlen)
    write_json(OUT / "zeitreihen.json", zeitreihen)
    for f in ("meta.json", "kennzahlen.json", "zeitreihen.json"):
        print(f"{f}: {(OUT / f).stat().st_size / 1024:.0f} KB")
    print(f"Gespeichert in {OUT} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
