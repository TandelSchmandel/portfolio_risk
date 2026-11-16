"""Sauberes Datenmodell: Kurse -> tägliche Renditen -> Kovarianzmatrix.

Alle Größen sind pro Handelstag, außer wenn ausdrücklich annualisiert. Die Spaltenreihenfolge
ist überall dieselbe (die der Ticker-Liste), Gewichte lassen sich daher direkt anwenden.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_risk.data import load_prices, log_returns, simple_returns

TRADING_DAYS = 252


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Stichprobenkovarianz (ddof = 1) der täglichen Renditen, beschriftet mit den Tickern."""
    if returns.isna().any().any():
        raise ValueError("Renditen enthalten fehlende Werte")
    if len(returns) < returns.shape[1] + 1:
        raise ValueError("Zu wenige Beobachtungen für eine invertierbare Kovarianzmatrix")
    return returns.cov()


def shrunk_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Kovarianz mit Ledoit-Wolf-Schrumpfung zur skalierten Einheitsmatrix (Ledoit und Wolf 2004).

    Die Stichprobenkovarianz hat bei 10 Titeln und etwa 1 250 Tagen kaum Rauschen, die
    Schrumpfung ist hier klein. Sie ist als Gegenprobe gedacht: weicht sie stark ab, ist die
    Schätzung instabil. Gibt (Matrix, Schrumpfungsintensität zwischen 0 und 1) zurück.
    """
    x = returns.to_numpy()
    n, p = x.shape
    centered = x - x.mean(axis=0)
    s = centered.T @ centered / n  # Maximum-Likelihood-Variante, wie im Originalaufsatz
    mu = np.trace(s) / p
    d2 = np.sum((s - mu * np.eye(p)) ** 2) / p
    b2 = sum(np.sum((np.outer(r, r) - s) ** 2) for r in centered) / (n**2 * p)
    b2 = min(b2, d2)
    shrinkage = 0.0 if d2 == 0 else b2 / d2
    shrunk = shrinkage * mu * np.eye(p) + (1 - shrinkage) * s
    shrunk *= n / (n - 1)  # auf ddof = 1 umrechnen, damit beide Matrizen vergleichbar sind
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns), float(shrinkage)


def is_positive_definite(matrix: pd.DataFrame | np.ndarray) -> bool:
    """True, wenn die symmetrische Matrix positiv definit ist (Cholesky-Zerlegung gelingt)."""
    try:
        np.linalg.cholesky(np.asarray(matrix, dtype=float))
    except np.linalg.LinAlgError:
        return False
    return True


@dataclass(frozen=True)
class RiskDataset:
    """Alle Eingangsdaten für die Risikoberechnung an einem Ort.

    prices:       bereinigte Schlusskurse (Adj Close), nur Tage, an denen alle Titel gehandelt wurden
    log_returns:  ln(P_t / P_(t-1)), einfach additiv über die Zeit
    simple_returns: P_t / P_(t-1) - 1, additiv über Titel (für Portfolio-Renditen)
    cov:          tägliche Kovarianzmatrix der **Log-Renditen** (Stichprobe, ddof = 1)
    """

    prices: pd.DataFrame
    log_returns: pd.DataFrame
    simple_returns: pd.DataFrame
    cov: pd.DataFrame

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def corr(self) -> pd.DataFrame:
        return self.log_returns.corr()

    @property
    def mean_daily(self) -> pd.Series:
        return self.log_returns.mean()

    @property
    def vol_daily(self) -> pd.Series:
        return self.log_returns.std()

    @property
    def vol_annual(self) -> pd.Series:
        return self.vol_daily * np.sqrt(TRADING_DAYS)

    @property
    def cov_annual(self) -> pd.DataFrame:
        return self.cov * TRADING_DAYS

    def portfolio_volatility(self, weights) -> float:
        """Tägliche Volatilität eines Portfolios mit festen Gewichten: sqrt(w' Cov w) (Näherung über Log-Renditen)."""
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != len(self.tickers):
            raise ValueError(f"{w.shape[0]} Gewichte für {len(self.tickers)} Ticker")
        return float(np.sqrt(w @ self.cov.to_numpy() @ w))

    def summary(self) -> dict:
        """Kennzahlen zur Dokumentation und zur schnellen Plausibilitätsprüfung."""
        eig = np.linalg.eigvalsh(self.cov.to_numpy())
        return {
            "Erster Renditetag": self.log_returns.index[0].date().isoformat(),
            "Letzter Renditetag": self.log_returns.index[-1].date().isoformat(),
            "Tagesrenditen": len(self.log_returns),
            "Ticker": len(self.tickers),
            "Positiv definit": is_positive_definite(self.cov),
            "Kleinster Eigenwert": float(eig[0]),
            "Konditionszahl": float(eig[-1] / eig[0]),
            "Mittlere Korrelation": float(self.corr.to_numpy()[np.triu_indices(len(self.tickers), 1)].mean()),
        }


def build_dataset(directory: str | Path, tickers: list[str] | None = None) -> RiskDataset:
    """Lädt die Kurse, wählt (in fester Reihenfolge) die Ticker und berechnet Renditen und Kovarianz."""
    prices = load_prices(directory)
    if tickers is not None:
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"Keine Kursdaten für {missing}")
        prices = prices[tickers]
    if (prices <= 0).any().any():
        raise ValueError("Kurse müssen positiv sein")
    lr = log_returns(prices)
    sr = simple_returns(prices)
    return RiskDataset(prices=prices, log_returns=lr, simple_returns=sr, cov=sample_covariance(lr))
