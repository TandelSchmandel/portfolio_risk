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
