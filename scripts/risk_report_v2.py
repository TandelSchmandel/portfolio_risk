"""Risikobericht Schritt 2, zweiter Durchlauf mit höherer Genauigkeit.

Schreibt nach reports/v2/:
  datenqualitaet.csv   Prüfung der Kursdaten (Lücken, Ausreißer, Split-Verdacht)
  datenbasis.csv       Kennzahlen des Datenmoduls und Konsistenzprüfung Kovarianz <-> Simulation
  var_modelle.csv      VaR und ES je Portfolio, Modell, Konfidenz und Horizont (1 und 10 Tage), mit KI
  backtest.csv         Ausnahmen, Kupiec- und Christoffersen-Test je Modell
  stresstests.csv      historische, datengetriebene und hypothetische Szenarien
  zusammenfassung.csv  die wichtigsten Kennzahlen je Portfolio auf einen Blick

Alle Eingangsgrößen kommen aus ``build_dataset`` (src/portfolio_risk/dataset.py): dieselben Kurse,
Log- und einfachen Renditen und dieselbe Kovarianzmatrix für Simulation, Stresstests und Backtest.

Der erste Durchlauf (scripts/risk_report.py, reports/risk_summary.csv) bleibt unverändert als Vergleich.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from portfolio_risk import portfolio_returns  # noqa: E402
from portfolio_risk import backtest, models, stress  # noqa: E402
from portfolio_risk.dataset import RiskDataset, build_dataset, shrunk_covariance  # noqa: E402
from portfolio_risk.portfolios import PORTFOLIOS, TICKERS  # noqa: E402
from portfolio_risk.quality import check_price_directory, has_blocking_issues  # noqa: E402
from portfolio_risk.risk import var_es_with_ci  # noqa: E402

SEED = 20261116
N_SIMS = 1_000_000
HORIZONS = (1, 10)
CONFIDENCES = (0.95, 0.99)
OUT = ROOT / "reports" / "v2"


def var_table(ds: RiskDataset, nu: float) -> pd.DataFrame:
    lr, cov = ds.log_returns, ds.cov
    sims = {
        "Normal (MC)": lambda w, h: models.simulate_normal(lr, w, h, N_SIMS, SEED, cov=cov),
        "Student-t (MC)": lambda w, h: models.simulate_student_t(lr, w, h, N_SIMS, SEED, nu=nu, cov=cov),
        "Block-Bootstrap (MC)": lambda w, h: models.simulate_block_bootstrap(lr, w, h, N_SIMS, SEED),
        "FHS (EWMA, MC)": lambda w, h: models.simulate_filtered_historical(lr, w, h, N_SIMS, SEED),
    }
    rows = []
    for name, weights in PORTFOLIOS.items():
        for h in HORIZONS:
            hist = models.historical_horizon_returns(lr, weights, h)
            samples = {"Historisch": hist.to_numpy()}
            samples.update({model: f(weights, h) for model, f in sims.items()})
            for model, s in samples.items():
                n_eff = len(s) / h if model == "Historisch" else None
                for c in CONFIDENCES:
                    res = var_es_with_ci(s, c, effective_n=n_eff)
                    rows.append({"Portfolio": name, "Horizont (Tage)": h, "Konfidenz": c, "Modell": model, **res})
            # Zum Vergleich: die alte Wurzel-Zeit-Skalierung des historischen Tages-VaR
            if h > 1:
                one_day = models.historical_horizon_returns(lr, weights, 1).to_numpy()
                for c in CONFIDENCES:
                    v = var_es_with_ci(one_day, c)["VaR"] * np.sqrt(h)
                    rows.append({"Portfolio": name, "Horizont (Tage)": h, "Konfidenz": c, "Modell": "Historisch x Wurzel(10) (nur Vergleich)", "VaR": v})
    return pd.DataFrame(rows)


def backtest_table(daily: dict[str, pd.Series]) -> pd.DataFrame:
    frames = []
    for name, r in daily.items():
        for c in CONFIDENCES:
            res = backtest.evaluate(backtest.rolling_var_forecasts(r, c), c)
            res.insert(0, "Konfidenz", c)
            res.insert(0, "Portfolio", name)
            frames.append(res.reset_index())
    return pd.concat(frames, ignore_index=True)


def stress_table(ds: RiskDataset, daily: dict[str, pd.Series], nu: float) -> pd.DataFrame:
    lr, cov = ds.log_returns, ds.cov
    rows = []

    def add(group, scenario, portfolio, value, detail=""):
        rows.append({"Gruppe": group, "Szenario": scenario, "Portfolio": portfolio, "Wert": value, "Detail": detail})

    for name, w in PORTFOLIOS.items():
        for sc in stress.HISTORICAL_SCENARIOS:
            add("1 Historisch", sc.name, name, stress.scenario_return(lr, w, sc.start, sc.end), f"{sc.start} bis {sc.end}: {sc.description}")

        for days in (1, 5, 20, 60):
            ww = stress.worst_window(lr, w, days)
            add("2 Datengetrieben", f"Schlechtestes {days}-Tage-Fenster", name, ww["Rendite"], f"{ww['Start']} bis {ww['Ende']}")
        dd = stress.max_drawdown(lr, w)
        add("2 Datengetrieben", "Maximaler Drawdown", name, dd["Drawdown"], f"Hoch {dd['Hoch']}, Tief {dd['Tief']}, erholt {dd['Erholt am']}")
        svar = stress.stressed_historical_var(daily[name], 0.99)
        add("2 Datengetrieben", "Stressed VaR 99 % 1 Tag (ungünstigstes 250-Tage-Fenster)", name, svar["VaR"], f"{svar['Start']} bis {svar['Ende']}, Verlust positiv")

        add("3 Hypothetisch", "Alle Titel -20 % (Vergleich v1)", name, stress.sector_shock(TICKERS, w, {}, -0.20))
        add("3 Hypothetisch", "Chips -30 %, Plattformen -15 %", name, stress.sector_shock(TICKERS, w, {**{t: -0.30 for t in stress.SEMIS}, **{t: -0.15 for t in stress.PLATFORMS}}))
        add("3 Hypothetisch", "KI-Blase platzt: Chips -50 %, Plattformen -30 %", name, stress.sector_shock(TICKERS, w, {**{t: -0.50 for t in stress.SEMIS}, **{t: -0.30 for t in stress.PLATFORMS}}))
        for label, basket, shock in [
            ("NVDA -30 %, Rest korreliert", {"NVDA": 1.0}, -0.30),
            ("Chip-Korb -30 %, Rest korreliert", {t: 1.0 for t in stress.SEMIS}, -0.30),
            ("Plattform-Korb -20 %, Rest korreliert", {t: 1.0 for t in stress.PLATFORMS}, -0.20),
        ]:
            cs = stress.conditional_shock(lr, w, basket, shock, cov=cov)
            detail = ", ".join(f"{k.split()[-1]} {v:+.1%}" for k, v in cs.items() if k.startswith("Bewegung"))
            add("3 Hypothetisch", label, name, cs["Rendite"], detail)
        base = stress.stressed_parametric_var(lr, w, 0.99, nu=nu, cov=cov)
        add("3 Hypothetisch", "Student-t-VaR 99 % 1 Tag, Basis", name, base, "Verlust positiv")
        add("3 Hypothetisch", "Student-t-VaR 99 % 1 Tag, alle Korrelationen 0.9", name, stress.stressed_parametric_var(lr, w, 0.99, corr=0.9, nu=nu, cov=cov), "Diversifikation bricht weg; Verlust positiv")
        add("3 Hypothetisch", "Student-t-VaR 99 % 1 Tag, Volatilität x1.5", name, stress.stressed_parametric_var(lr, w, 0.99, vol_factor=1.5, nu=nu, cov=cov), "Verlust positiv")
    return pd.DataFrame(rows)


def dataset_table(ds: RiskDataset, nu: float) -> pd.DataFrame:
    """Kennzahlen des Datenmoduls und eine Konsistenzprüfung je Portfolio.

    Die Normal-Simulation muss dieselbe Tagesvolatilität liefern wie sqrt(w' Cov w) aus dem
    Datenmodul (bis auf Monte-Carlo-Fehler und die kleine Umrechnung Log- in einfache Rendite).
    """
    summary = ds.summary()
    _, shrink = shrunk_covariance(ds.log_returns)
    rows = {k: {name: v for name in PORTFOLIOS} for k, v in summary.items()}
    rows["Ledoit-Wolf-Schrumpfung"] = {name: shrink for name in PORTFOLIOS}
    rows["Student-t nu"] = {name: nu for name in PORTFOLIOS}
    for name, w in PORTFOLIOS.items():
        sigma = ds.portfolio_volatility(w)
        sim = models.simulate_normal(ds.log_returns, w, 1, 200_000, SEED, cov=ds.cov)
        rows.setdefault("Tagesvolatilität sqrt(w' Cov w)", {})[name] = sigma
        rows.setdefault("Volatilität p. a.", {})[name] = sigma * np.sqrt(252)
        rows.setdefault("Tagesvolatilität Normal-MC (Prüfung)", {})[name] = float(sim.std(ddof=1))
    return pd.DataFrame(rows).T


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    quality = check_price_directory(ROOT / "data" / "raw")
    quality.to_csv(OUT / "datenqualitaet.csv", float_format="%.4f")
    if has_blocking_issues(quality):
        raise SystemExit("Datenfehler gefunden, siehe reports/v2/datenqualitaet.csv")

    ds = build_dataset(ROOT / "data" / "raw", TICKERS)
    daily = {name: portfolio_returns(ds.simple_returns, w) for name, w in PORTFOLIOS.items()}
    nu = round(models.fit_t_degrees_of_freedom(ds.log_returns, cov=ds.cov), 2)
    print(f"Daten: {ds.log_returns.index[0].date()} bis {ds.log_returns.index[-1].date()}, {len(ds.log_returns)} Tagesrenditen, Student-t nu = {nu}")
    basis = dataset_table(ds, nu)
    basis.to_csv(OUT / "datenbasis.csv", float_format="%.6f")

    var = var_table(ds, nu)
    var.to_csv(OUT / "var_modelle.csv", index=False, float_format="%.5f")
    print(f"VaR-Modelle fertig ({time.time() - t0:.0f} s)")

    bt = backtest_table(daily)
    bt.to_csv(OUT / "backtest.csv", index=False, float_format="%.4f")
    print(f"Backtest fertig ({time.time() - t0:.0f} s)")

    st = stress_table(ds, daily, nu)
    st.to_csv(OUT / "stresstests.csv", index=False, float_format="%.4f")

    summary = {}
    for name in PORTFOLIOS:
        v = var[var["Portfolio"] == name]

        def pick(model, h, c, col="VaR"):
            return float(v[(v["Modell"] == model) & (v["Horizont (Tage)"] == h) & (v["Konfidenz"] == c)][col].iloc[0])

        s = st[st["Portfolio"] == name].set_index("Szenario")["Wert"]
        b = bt[(bt["Portfolio"] == name) & (bt["Konfidenz"] == 0.99)].set_index("Modell")
        summary[name] = {
            "VaR 99 % 1T historisch": pick("Historisch", 1, 0.99),
            "VaR 99 % 1T Normal (wie v1)": pick("Normal (MC)", 1, 0.99),
            "VaR 99 % 1T Student-t": pick("Student-t (MC)", 1, 0.99),
            "VaR 99 % 1T Bootstrap": pick("Block-Bootstrap (MC)", 1, 0.99),
            "VaR 99 % 1T FHS (aktuelle Vola)": pick("FHS (EWMA, MC)", 1, 0.99),
            "ES 99 % 1T Student-t": pick("Student-t (MC)", 1, 0.99, "ES"),
            "VaR 99 % 10T historisch": pick("Historisch", 10, 0.99),
            "VaR 99 % 10T Student-t": pick("Student-t (MC)", 10, 0.99),
            "VaR 99 % 10T FHS": pick("FHS (EWMA, MC)", 10, 0.99),
            "Backtest 99 %: Ausnahmen Normal": int(b.loc["Normal", "Ausnahmen"]),
            "Backtest 99 %: Ausnahmen Student-t": int(b.loc["Student-t", "Ausnahmen"]),
            "Backtest 99 %: Ausnahmen FHS": int(b.loc["FHS (EWMA)", "Ausnahmen"]),
            "Backtest 99 %: erwartet": float(b.loc["Normal", "Erwartet"]),
            "Maximaler Drawdown": s["Maximaler Drawdown"],
            "Schlechtester Tag": s["Schlechtestes 1-Tage-Fenster"],
            "KI-Blase platzt (Chips -50 %, Plattformen -30 %)": s["KI-Blase platzt: Chips -50 %, Plattformen -30 %"],
        }
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "zusammenfassung.csv", float_format="%.4f")
    pd.set_option("display.width", 200)
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nGespeichert in {OUT} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
