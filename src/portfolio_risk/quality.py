"""Prüfung der Kursdaten vor der Risikoberechnung.

Die Prüfung entfernt nichts. Sie meldet Auffälligkeiten, damit man entscheiden kann,
ob ein Wert ein Datenfehler ist (z. B. ein nicht bereinigter Aktiensplit) oder eine
echte Marktbewegung (z. B. ein Kurssprung nach Quartalszahlen). Echte Extremtage
gehören in die Risikoberechnung, weil genau sie den VaR treiben.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Ab diesem robusten z-Wert gilt eine Tagesrendite als Ausreißer (Median/MAD statt Mittelwert/Std).
ROBUST_Z_THRESHOLD = 8.0
# Eine Tagesbewegung von mehr als 40 % (log) ist bei diesen Titeln fast immer ein Datenfehler.
SPLIT_SUSPECT_LOG_RETURN = 0.40
# Lücken zwischen zwei Handelstagen von mehr als 5 Kalendertagen sind ungewöhnlich (Feiertage + Wochenende = max. 4).
MAX_CALENDAR_GAP_DAYS = 5


def robust_z(values: pd.Series) -> pd.Series:
    """Robuster z-Wert: (x - Median) / (1.4826 * MAD). Unempfindlich gegen die Ausreißer selbst."""
    median = values.median()
    mad = (values - median).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=values.index)
    return (values - median) / (1.4826 * mad)


def check_price_file(path: str | Path) -> dict:
    """Prüft eine Kurs-CSV und gibt eine Zeile mit Kennzahlen und gefundenen Auffälligkeiten zurück."""
    path = Path(path)
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    ticker = path.stem.upper()
    adj = df.set_index("Date")["Adj Close"].astype(float)
    log_ret = np.log(adj / adj.shift(1)).dropna()
    z = robust_z(log_ret)

    gaps = adj.index.to_series().diff().dt.days
    row = {
        "Ticker": ticker,
        "Erster Tag": adj.index.min().date().isoformat(),
        "Letzter Tag": adj.index.max().date().isoformat(),
        "Handelstage": len(adj),
        "Fehlende Werte": int(df[["Adj Close"]].isna().sum().sum()),
        "Doppelte Daten": int(df["Date"].duplicated().sum()),
        "Kurse <= 0": int((adj <= 0).sum()),
        "Lücken > 5 Kalendertage": int((gaps > MAX_CALENDAR_GAP_DAYS).sum()),
        "Größter Tagesgewinn (log)": float(log_ret.max()),
        "Größter Tagesverlust (log)": float(log_ret.min()),
        "Ausreißer |robust z| > 8": int((z.abs() > ROBUST_Z_THRESHOLD).sum()),
        "Ausreißer-Tage": ";".join(d.date().isoformat() for d in log_ret.index[z.abs() > ROBUST_Z_THRESHOLD]),
        "Split-Verdacht": int((log_ret.abs() > SPLIT_SUSPECT_LOG_RETURN).sum()),
    }
    if "Close" in df.columns:
        # Weicht das Verhältnis Adj Close / Close sprunghaft ab, ist eine Bereinigung inkonsistent.
        ratio = (df.set_index("Date")["Adj Close"] / df.set_index("Date")["Close"]).astype(float)
        row["Max. Sprung Adj/Close-Verhältnis"] = float(ratio.pct_change().abs().max())
    return row


def check_price_directory(directory: str | Path) -> pd.DataFrame:
    """Prüft alle CSV-Dateien eines Ordners."""
    files = sorted(Path(directory).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Keine CSV-Dateien in {directory}")
    return pd.DataFrame([check_price_file(f) for f in files]).set_index("Ticker")


def has_blocking_issues(report: pd.DataFrame) -> bool:
    """True, wenn Fehler vorliegen, die eine Berechnung verfälschen würden (nicht bloß Extremtage)."""
    blocking = ["Fehlende Werte", "Doppelte Daten", "Kurse <= 0", "Split-Verdacht"]
    return bool((report[blocking] > 0).any().any())
