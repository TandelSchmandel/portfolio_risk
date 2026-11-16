"""Stresstests in drei klar getrennten Gruppen.

1. **Historische Szenarien**: echte Episoden aus den Daten mit festen Start- und Enddaten.
   Die Rendite ist Buy-and-hold vom Schlusskurs vor dem Starttag bis zum Schlusskurs am Endtag.
2. **Datengetriebene Extremwerte**: schlechteste Fenster über 1, 5, 20 und 60 Tage sowie der
   maximale Drawdown. Sie hängen nicht von einer Auswahl der Ereignisse ab.
3. **Hypothetische Szenarien**: Schocks, die es so in den Daten nicht gab. Pauschale Schocks
   (alle Titel gleich), Sektorschocks, korrelierte Schocks (ein Titel oder Korb fällt, die
   übrigen bewegen sich gemäß ihrer historischen Kovarianz mit) sowie Parameterstress
   (Korrelationen oder Volatilitäten erhöht).

Alle Werte sind Portfolio-Renditen (negativ = Verlust), außer den VaR-Größen in Gruppe 3,
die wie üblich als positiver Verlust angegeben werden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from portfolio_risk.models import resolve_covariance, horizon_portfolio_return

SEMIS = ["NVDA", "AVGO", "TSM", "AMD", "MU"]
PLATFORMS = ["MSFT", "GOOGL", "AMZN", "META", "PLTR"]


@dataclass(frozen=True)
class Scenario:
    name: str
    start: str
    end: str
    description: str


HISTORICAL_SCENARIOS = [
    Scenario("Zinswende 2022 (Vergleich v1)", "2022-01-03", "2022-10-14", "Fed erhöht Zinsen, Tech-Bewertungen fallen; gleicher Zeitraum wie im ersten Durchlauf"),
    Scenario("META-Gewinnwarnung", "2022-02-03", "2022-02-03", "Einzeltag: META -26 % nach Quartalszahlen"),
    Scenario("Ausverkauf Q2 2022", "2022-04-01", "2022-06-16", "Inflationsschock, Nasdaq-Tief Mitte Juni"),
    Scenario("Rotation und Yen-Carry Sommer 2024", "2024-07-11", "2024-08-05", "Rotation aus Mega-Caps, Auflösung des Yen-Carry-Trades"),
    Scenario("DeepSeek-Schock", "2025-01-27", "2025-01-27", "Einzeltag: Zweifel am KI-Investitionsbedarf, NVDA -17 %"),
    Scenario("Zollschock April 2025", "2025-04-03", "2025-04-08", "Vier Handelstage nach Ankündigung der US-Zölle"),
]


def scenario_return(log_returns: pd.DataFrame, weights, start: str, end: str) -> float:
    """Buy-and-hold-Rendite des Portfolios im Zeitraum (inklusive Start- und Endtag)."""
    window = log_returns.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if window.empty:
        raise ValueError(f"Keine Daten im Zeitraum {start} bis {end}")
    return float(horizon_portfolio_return(window.sum().to_numpy(), np.asarray(weights, dtype=float)))


def worst_window(log_returns: pd.DataFrame, weights, days: int) -> dict:
    """Schlechtestes Buy-and-hold-Fenster über `days` Handelstage mit Start- und Enddatum."""
    cum = log_returns.rolling(days).sum().dropna()
    ret = pd.Series(horizon_portfolio_return(cum.to_numpy(), np.asarray(weights, dtype=float)), index=cum.index)
    end = ret.idxmin()
    start = log_returns.index[log_returns.index.get_loc(end) - days + 1]
    return {"Rendite": float(ret.loc[end]), "Start": start.date().isoformat(), "Ende": end.date().isoformat()}


def max_drawdown(log_returns: pd.DataFrame, weights) -> dict:
    """Maximaler Rückgang vom Höchststand (Buy-and-hold ab dem ersten Tag), mit Hoch, Tief und Erholung."""
    w = np.asarray(weights, dtype=float)
    value = pd.Series(np.exp(log_returns.cumsum()).to_numpy() @ w, index=log_returns.index)
    value = pd.concat([pd.Series([1.0], index=[log_returns.index[0] - pd.Timedelta(days=1)]), value])
    running_max = value.cummax()
    dd = value / running_max - 1
    trough = dd.idxmin()
    peak = value.loc[:trough].idxmax()
    after = value.loc[trough:]
    recovered = after[after >= value.loc[peak]]
    return {
        "Drawdown": float(dd.loc[trough]),
        "Hoch": peak.date().isoformat(),
        "Tief": trough.date().isoformat(),
        "Erholt am": recovered.index[0].date().isoformat() if not recovered.empty else "noch nicht",
    }


def sector_shock(tickers: list[str], weights, shocks: dict[str, float], default: float = 0.0) -> float:
    """Hypothetischer Schock mit eigener Kursänderung je Titel, z. B. {'NVDA': -0.3}."""
    w = np.asarray(weights, dtype=float)
    moves = np.array([shocks.get(t, default) for t in tickers])
    return float(w @ moves)


def conditional_shock(log_returns: pd.DataFrame, weights, basket: dict[str, float], shock: float, cov=None) -> dict:
    """Korrelierter Schock: ein Korb (z. B. NVDA allein oder alle Chiptitel) fällt um `shock`.

    Die übrigen Titel bewegen sich um ihren bedingten Erwartungswert unter der historischen
    Kovarianz: E[r | a'r = s] = Cov * a / (a' Cov a) * s, gerechnet in Log-Renditen.
    Das ist die Bewegung, die man im Mittel beobachten würde, wenn der Korb so stark fällt.
    """
    tickers = list(log_returns.columns)
    a = np.array([basket.get(t, 0.0) for t in tickers])
    a = a / a.sum()
    cov = resolve_covariance(log_returns, cov)
    s = np.log1p(shock)
    moves = cov @ a / (a @ cov @ a) * s
    w = np.asarray(weights, dtype=float)
    result = {"Rendite": float(horizon_portfolio_return(moves, w))}
    result.update({f"Bewegung {t}": float(np.expm1(m)) for t, m in zip(tickers, moves)})
    return result


def stressed_parametric_var(log_returns: pd.DataFrame, weights, confidence: float = 0.99, corr: float | None = None, vol_factor: float = 1.0, nu: float | None = None, cov=None) -> float:
    """Parametrischer Tages-VaR (Normal oder Student-t, ohne Drift) mit gestressten Parametern.

    corr: alle paarweisen Korrelationen auf diesen Wert setzen (z. B. 0.9: Diversifikation bricht weg).
    cov: Basis-Kovarianz (z. B. ``RiskDataset.cov``); ohne Angabe die Stichprobe der Renditen.
    vol_factor: alle Volatilitäten mit diesem Faktor multiplizieren (z. B. 1.5).
    """
    base = resolve_covariance(log_returns, cov)
    sd = np.sqrt(np.diag(base))
    vol = sd * vol_factor
    c = base / np.outer(sd, sd)
    if corr is not None:
        c = np.full_like(c, corr)
        np.fill_diagonal(c, 1.0)
    cov = np.outer(vol, vol) * c
    w = np.asarray(weights, dtype=float)
    sigma_p = float(np.sqrt(w @ cov @ w))
    if nu is None:
        q = stats.norm.ppf(1 - confidence) * sigma_p
    else:
        q = stats.t.ppf(1 - confidence, nu) * sigma_p * np.sqrt((nu - 2) / nu)
    # Quantil der Log-Rendite in einfache Rendite umrechnen, damit es mit den Simulationen vergleichbar ist.
    return float(-np.expm1(q))


def stressed_historical_var(portfolio_returns: pd.Series, confidence: float = 0.99, window: int = 250) -> dict:
    """Stressed VaR nach Basel 2.5: historischer VaR im ungünstigsten 250-Tage-Fenster der Daten."""
    rolling = (-portfolio_returns.rolling(window).quantile(1 - confidence)).dropna()
    end = rolling.idxmax()
    start = portfolio_returns.index[portfolio_returns.index.get_loc(end) - window + 1]
    return {"VaR": float(rolling.loc[end]), "Start": start.date().isoformat(), "Ende": end.date().isoformat()}
