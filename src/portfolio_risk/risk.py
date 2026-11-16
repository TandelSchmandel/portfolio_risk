"""Risikokennzahlen: historischer und Monte-Carlo-VaR, Expected Shortfall und Stresstests.

Konvention: Verluste werden als positive Zahl angegeben, z. B. bedeutet ein VaR von 0.03
einen Verlust von 3 % des Portfolios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_confidence(confidence: float) -> None:
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"Konfidenzniveau muss zwischen 0.5 und 1.0 liegen, nicht {confidence}")


def _validate_weights(weights, n_assets: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != n_assets:
        raise ValueError(f"{w.shape[0]} Gewichte für {n_assets} Ticker")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"Gewichte summieren sich zu {w.sum():.4f}, erwartet 1.0")
    return w


def historical_var(portfolio_returns, confidence: float = 0.95) -> float:
    """Historischer VaR: Verlust, der mit der Wahrscheinlichkeit `confidence` nicht überschritten wird."""
    _check_confidence(confidence)
    r = np.asarray(portfolio_returns, dtype=float)
    return float(-np.quantile(r, 1 - confidence))


def expected_shortfall(portfolio_returns, confidence: float = 0.95) -> float:
    """Expected Shortfall: durchschnittlicher Verlust in den schlechtesten Fällen jenseits des VaR."""
    _check_confidence(confidence)
    r = np.asarray(portfolio_returns, dtype=float)
    cutoff = np.quantile(r, 1 - confidence)
    tail = r[r <= cutoff]
    return float(-tail.mean())


def monte_carlo_var(
    asset_returns: pd.DataFrame,
    weights,
    confidence: float = 0.95,
    n_sims: int = 100_000,
    seed: int = 42,
) -> float:
    """Monte-Carlo-VaR für einen Tag.

    Annahme: Tagesrenditen sind multivariat normalverteilt mit dem historischen
    Mittelwert und der historischen Kovarianzmatrix. Fat Tails werden damit unterschätzt.
    """
    _check_confidence(confidence)
    w = _validate_weights(weights, asset_returns.shape[1])
    mu = asset_returns.mean().to_numpy()
    cov = asset_returns.cov().to_numpy()
    rng = np.random.default_rng(seed)
    simulated = rng.multivariate_normal(mu, cov, size=n_sims, method="cholesky")
    return float(-np.quantile(simulated @ w, 1 - confidence))


def replay_return(asset_returns: pd.DataFrame, weights, start: str, end: str) -> float:
    """Stresstest durch Historienwiedergabe: kumulierte Portfolio-Rendite im Zeitraum start bis end."""
    w = _validate_weights(weights, asset_returns.shape[1])
    window = asset_returns.loc[start:end]
    if window.empty:
        raise ValueError(f"Keine Daten im Zeitraum {start} bis {end}")
    daily = window.to_numpy() @ w
    return float(np.prod(1 + daily) - 1)


def worst_window(asset_returns: pd.DataFrame, weights, days: int = 20) -> tuple[float, pd.Timestamp]:
    """Schlechtestes zusammenhängendes Fenster von `days` Handelstagen im Datensatz.

    Gibt die kumulierte Rendite und das Enddatum des Fensters zurück.
    """
    w = _validate_weights(weights, asset_returns.shape[1])
    daily = pd.Series(asset_returns.to_numpy() @ w, index=asset_returns.index)
    cumulative = (1 + daily).rolling(days).apply(np.prod, raw=True) - 1
    end = cumulative.idxmin()
    return float(cumulative.loc[end]), end


def uniform_shock(weights, shock: float) -> float:
    """Hypothetischer Schock: alle Titel fallen um denselben Prozentsatz, z. B. -0.20."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() * shock)


# --------------------------------------------------------------------------- VaR und ES mit Konfidenzintervall


def var_es_with_ci(samples, confidence: float = 0.99, ci_level: float = 0.95, effective_n: float | None = None) -> dict:
    """VaR und ES aus einer Stichprobe von Renditen, mit verteilungsfreiem Konfidenzintervall des VaR.

    Das Intervall kommt aus der Binomialverteilung der Ordnungsstatistiken: Die Anzahl der
    Werte unter dem wahren Quantil ist Binomial(n, 1 - confidence). Es setzt nur unabhängige
    Ziehungen voraus, keine Verteilungsannahme.

    * Bei Monte-Carlo-Stichproben misst es den **Simulationsfehler** (wird mit mehr Läufen kleiner).
    * Bei historischen Daten misst es den **Stichprobenfehler** der Daten (wird nur mit mehr Daten kleiner).
    * Bei überlappenden Mehrtagesfenstern sind die Werte abhängig; dann `effective_n`
      (z. B. n / Horizont) angeben, sonst ist das Intervall zu schmal.

    Verluste sind positiv. ci_low/ci_high sind die Grenzen des VaR-Intervalls.
    """
    from scipy import stats

    _check_confidence(confidence)
    r = np.asarray(samples, dtype=float)
    n = r.size
    p = 1 - confidence
    q = np.quantile(r, p)
    tail = r[r <= q]

    n_eff = n if effective_n is None else float(effective_n)
    alpha = 1 - ci_level
    k_low = stats.binom.ppf(alpha / 2, int(round(n_eff)), p)
    k_high = stats.binom.ppf(1 - alpha / 2, int(round(n_eff)), p) + 1
    p_low = np.clip(k_low / n_eff, 0.0, 1.0)
    p_high = np.clip(k_high / n_eff, 0.0, 1.0)
    # Ein kleineres Quantil der Rendite ist ein größerer Verlust: Grenzen tauschen.
    return {
        "VaR": float(-q),
        "VaR KI unten": float(-np.quantile(r, p_high)),
        "VaR KI oben": float(-np.quantile(r, p_low)),
        "ES": float(-tail.mean()),
        "n": n,
    }
