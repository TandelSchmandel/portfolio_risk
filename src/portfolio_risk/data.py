"""Laden und Vorbereiten historischer Kursdaten.

Erwartetes Format: eine CSV-Datei pro Wertpapier mit den Spalten
``Date`` und ``Adj Close`` (das Format des Yahoo-Finance-Downloads).
Der Dateiname ohne Endung dient als Ticker, z. B. ``AAPL.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("Date", "Adj Close")


def load_price_csv(path: str | Path) -> pd.Series:
    """Liest eine Kurs-CSV und gibt die bereinigten Schlusskurse als Zeitreihe zurück."""
    path = Path(path)
    df = pd.read_csv(path, parse_dates=["Date"])
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: fehlende Spalten {missing}")

    series = df.set_index("Date")["Adj Close"].astype(float).sort_index()
    # Doppelte Datumseinträge kommen in Downloads vor; der letzte Eintrag gewinnt.
    return series[~series.index.duplicated(keep="last")]


def load_prices(directory: str | Path) -> pd.DataFrame:
    """Lädt alle CSV-Dateien eines Ordners zu einer Tabelle mit einer Spalte pro Ticker.

    Es bleiben nur Handelstage übrig, an denen für alle Ticker ein Kurs vorliegt.
    So sind Renditen und Kovarianzen über dieselben Zeiträume berechnet.
    """
    directory = Path(directory)
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Keine CSV-Dateien in {directory}")

    columns = {file.stem.upper(): load_price_csv(file) for file in files}
    prices = pd.DataFrame(columns).sort_index()
    return prices.dropna()


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Tägliche einfache Renditen: P_t / P_(t-1) - 1."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Tägliche logarithmische Renditen: ln(P_t / P_(t-1))."""
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_returns(returns: pd.DataFrame, weights: list[float] | np.ndarray) -> pd.Series:
    """Renditen eines Portfolios mit festen Gewichten.

    Annahme: Das Portfolio wird täglich auf die Zielgewichte zurückgesetzt (tägliches Rebalancing).
    Die Gewichte müssen zur Anzahl der Ticker passen und sich zu 1 summieren.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != returns.shape[1]:
        raise ValueError(f"{w.shape[0]} Gewichte für {returns.shape[1]} Ticker")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"Gewichte summieren sich zu {w.sum():.4f}, erwartet 1.0")
    return pd.Series(returns.to_numpy() @ w, index=returns.index, name="portfolio")
