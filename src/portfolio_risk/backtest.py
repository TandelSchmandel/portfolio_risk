"""VaR-Backtesting: Prognosen außerhalb der Stichprobe erstellen und mit den echten Verlusten vergleichen.

Ablauf: An jedem Tag t wird der VaR für Tag t+1 nur aus den Daten bis t geschätzt
(rollierendes Fenster). Eine **Ausnahme** liegt vor, wenn der echte Verlust an t+1 größer ist
als der prognostizierte VaR. Ein gutes Modell hat bei 99 % etwa 1 % Ausnahmen, und die
Ausnahmen treten nicht gehäuft hintereinander auf.

Getestet wird auf Ebene der Portfolio-Tagesrendite (einfache Rendite, tägliches Rebalancing).
Für die Student-t ist das exakt dasselbe Modell wie die multivariate Simulation, denn eine
Linearkombination einer multivariaten t-Verteilung ist wieder t-verteilt mit gleichem nu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from portfolio_risk.models import EWMA_LAMBDA

WINDOW = 250  # ein Handelsjahr, wie im Basler Rahmenwerk


# --------------------------------------------------------------------------- Prognosemodelle (Portfolio-Ebene)


def _t_nu_univariate(x: np.ndarray, grid=np.arange(2.1, 50.01, 0.1)) -> float:
    """ML-Schätzung von nu bei fester Varianz (Formparameter so skaliert, dass Var = Stichprobenvarianz)."""
    s = x.std(ddof=1)
    nu = np.asarray(grid, dtype=float)[:, None]
    scale = s * np.sqrt((nu - 2) / nu)
    ll = stats.t.logpdf(x[None, :], nu, scale=scale).sum(axis=1)
    return float(grid[int(np.argmax(ll))])


def _forecast_historical(window: np.ndarray, p: float, _state) -> float:
    return -np.quantile(window, p)


def _forecast_normal(window: np.ndarray, p: float, _state) -> float:
    return -stats.norm.ppf(p) * window.std(ddof=1)


def _forecast_student_t(window: np.ndarray, p: float, _state) -> float:
    nu = _t_nu_univariate(window - window.mean())
    scale = window.std(ddof=1) * np.sqrt((nu - 2) / nu)
    return -stats.t.ppf(p, nu) * scale


def _forecast_ewma_normal(window: np.ndarray, p: float, state) -> float:
    return -stats.norm.ppf(p) * state["sigma_next"]


def _forecast_fhs(window: np.ndarray, p: float, state) -> float:
    z = state["resid"]
    return -np.quantile(z, p) * state["sigma_next"]


MODELS = {
    "Historisch": _forecast_historical,
    "Normal": _forecast_normal,
    "Student-t": _forecast_student_t,
    "EWMA-Normal": _forecast_ewma_normal,
    "FHS (EWMA)": _forecast_fhs,
}


def rolling_var_forecasts(returns: pd.Series, confidence: float = 0.99, window: int = WINDOW, lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    """Out-of-sample-VaR-Prognosen aller Modelle für jeden Tag nach dem ersten Fenster.

    Die Zeile mit Datum t enthält den VaR, der am Vortag für Tag t prognostiziert wurde,
    und den echten Verlust an Tag t.
    """
    r = returns.to_numpy(dtype=float)
    p = 1 - confidence
    # EWMA-Varianz über die gesamte Reihe; var[t] nutzt nur Daten bis t-1, ist also ohne Vorgriff.
    var = np.empty_like(r)
    var[0] = r[:30].var(ddof=1)
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    sigma = np.sqrt(var)

    rows = []
    for t in range(window, len(r)):
        hist = r[t - window : t]
        resid = (hist - hist.mean()) / sigma[t - window : t]
        state = {"sigma_next": sigma[t], "resid": resid / resid.std(ddof=1)}
        row = {name: f(hist, p, state) for name, f in MODELS.items()}
        row["Verlust"] = -r[t]
        rows.append(row)
    return pd.DataFrame(rows, index=returns.index[window:])


# --------------------------------------------------------------------------- Tests


def kupiec_pof(exceptions: int, n: int, confidence: float) -> tuple[float, float]:
    """Kupiec-Test (Proportion of Failures): stimmt die Ausnahmequote mit 1 - confidence überein?

    Gibt (LR-Statistik, p-Wert) zurück. LR ist unter H0 Chi2-verteilt mit 1 Freiheitsgrad.
    p-Wert < 0.05: Ausnahmequote weicht signifikant ab, Modell verwerfen.
    """
    p = 1 - confidence
    x = exceptions
    pi = x / n
    ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_alt = (n - x) * np.log(1 - pi) if x < n else 0.0
    ll_alt += x * np.log(pi) if x > 0 else 0.0
    lr = float(-2 * (ll_null - ll_alt))
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen-Test auf Unabhängigkeit: folgt eine Ausnahme gehäuft auf eine Ausnahme?

    Prüft, ob die Wahrscheinlichkeit einer Ausnahme vom Vortag abhängt (Markov-Kette 1. Ordnung).
    Gibt (LR-Statistik, p-Wert) zurück, Chi2 mit 1 Freiheitsgrad.
    """
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    def ll(count0: int, count1: int, prob: float) -> float:
        out = 0.0
        if count0:
            out += count0 * np.log(1 - prob)
        if count1:
            out += count1 * np.log(prob)
        return out

    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    if pi in (0.0, 1.0):
        return 0.0, 1.0
    ll_null = ll(n00 + n10, n01 + n11, pi)
    ll_alt = ll(n00, n01, pi0) + ll(n10, n11, pi1)
    lr = float(max(-2 * (ll_null - ll_alt), 0.0))
    return lr, float(stats.chi2.sf(lr, 1))


def basel_traffic_light(exceptions_last_250: int) -> str:
    """Basler Ampel für 99 %-VaR über 250 Tage: grün 0–4, gelb 5–9, rot ab 10 Ausnahmen."""
    if exceptions_last_250 <= 4:
        return "grün"
    if exceptions_last_250 <= 9:
        return "gelb"
    return "rot"


def evaluate(forecasts: pd.DataFrame, confidence: float) -> pd.DataFrame:
    """Zählt Ausnahmen je Modell und wendet Kupiec- und Christoffersen-Test an."""
    rows = []
    loss = forecasts["Verlust"].to_numpy()
    for model in MODELS:
        hits = loss > forecasts[model].to_numpy()
        n, x = hits.size, int(hits.sum())
        lr_pof, p_pof = kupiec_pof(x, n, confidence)
        lr_ind, p_ind = christoffersen_independence(hits)
        lr_cc = lr_pof + lr_ind
        row = {
            "Modell": model,
            "Tage": n,
            "Ausnahmen": x,
            "Erwartet": round(n * (1 - confidence), 1),
            "Quote": x / n,
            "Kupiec LR": lr_pof,
            "Kupiec p": p_pof,
            "Christoffersen Unabh. p": p_ind,
            "Bedingte Abdeckung p": float(stats.chi2.sf(lr_cc, 2)),
            "Mittlerer VaR": float(forecasts[model].mean()),
        }
        if np.isclose(confidence, 0.99):
            row["Ausnahmen letzte 250 Tage"] = int(hits[-250:].sum())
            row["Basler Ampel"] = basel_traffic_light(int(hits[-250:].sum()))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Modell")
