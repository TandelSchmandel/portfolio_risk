"""Kursdaten fortschreiben: tägliche Kurse von Yahoo Finance laden, prüfen und als CSV speichern.

Ablauf (``update_directory``):

1. Je Ticker die ganze Historie ab dem festen Starttag neu laden. Yahoo rechnet ``Adj Close`` bei jeder
   Dividende rückwirkend um; nur eine vollständige Reihe aus einem Abruf ist in sich stimmig.
   Bloßes Anhängen neuer Zeilen würde am Ex-Tag die Dividende aus der Rendite verlieren.
2. Den laufenden Handelstag verwerfen, solange die US-Börse nicht geschlossen hat.
3. Gegen die bisherigen Dateien prüfen: kein Tag darf fehlen, die Tagesrenditen der gemeinsamen Tage
   müssen übereinstimmen, alle Titel müssen am selben Tag enden.
4. Die Datenprüfung aus ``quality.py`` und ``build_dataset`` müssen durchlaufen.
5. Erst wenn alles stimmt, werden die Dateien ersetzt. Bei einem Fehler bleibt der alte Stand unverändert.

``yfinance`` wird nur hier gebraucht (``requirements-update.txt``), nicht für Rechnung oder Dienst.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from portfolio_risk.dataset import build_dataset
from portfolio_risk.quality import check_price_directory, has_blocking_issues

START = "2021-11-16"  # erster Kurstag des Datensatzes; bleibt fest, das Fenster wächst nach vorn
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]
COLUMNS = [*PRICE_COLUMNS, "Volume"]
PRICE_DECIMALS = 4
# Größte erlaubte Abweichung einer Tagesrendite (log) zwischen altem und neuem Stand. Rundung der alten
# Dateien (2 Nachkommastellen) macht bis etwa 0,2 % aus, eine nachgetragene Dividende bis etwa 0,5 %.
MAX_RETURN_DEVIATION = 0.01
NEW_YORK = ZoneInfo("America/New_York")
SESSION_FINAL = time(17, 0)  # US-Börsenschluss 16 Uhr, danach eine Stunde Puffer


class UpdateError(RuntimeError):
    """Die neuen Daten sind unvollständig oder unplausibel. Es wurde nichts geschrieben."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Kursdaten nicht aktualisiert:\n- " + "\n- ".join(problems))


@dataclass
class UpdateResult:
    changed: bool
    previous_last_day: str | None
    last_day: str
    new_days: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def last_complete_session(now: datetime | None = None) -> date:
    """Letzter Kalendertag, dessen US-Handel sicher abgeschlossen ist."""
    now = datetime.now(NEW_YORK) if now is None else now.astimezone(NEW_YORK)
    return now.date() if now.time() >= SESSION_FINAL else now.date() - timedelta(days=1)


def normalize_history(raw: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Bringt eine Kurstabelle von Yahoo (``yfinance``, ``auto_adjust=False``) ins CSV-Format des Projekts.

    Ergebnis: Index ``Date`` (aufsteigend, ohne Zeitzone), Spalten Open, High, Low, Close, Adj Close, Volume.
    Tage nach ``cutoff`` und Zeilen ohne Schlusskurs fallen weg.
    """
    if raw is None or len(raw) == 0:
        raise ValueError("keine Daten erhalten")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):  # yf.download liefert (Feld, Ticker)
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"fehlende Spalten {missing}")
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_localize(None)  # die Börsenzeit steht schon im lokalen Datum
    df.index = index.normalize().rename("Date")
    df = df[COLUMNS].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index.date <= cutoff]
    df = df.dropna(subset=["Close", "Adj Close"])
    if df.empty:
        raise ValueError("keine abgeschlossenen Handelstage erhalten")
    if df[PRICE_COLUMNS].isna().any().any() or (df[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("Kurse fehlen oder sind nicht positiv")
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].astype(float).round(PRICE_DECIMALS)
    df["Volume"] = df["Volume"].fillna(0).round().astype("int64")
    return df


def read_price_csv(path: str | Path) -> pd.DataFrame:
    """Liest eine Kursdatei des Projekts mit Datum als Index."""
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[~df.index.duplicated(keep="last")]


def write_price_csv(df: pd.DataFrame, path: str | Path) -> None:
    out = df.copy()
    out.index = out.index.strftime("%Y-%m-%d")
    out.to_csv(path, index_label="Date", float_format=f"%.{PRICE_DECIMALS}f", lineterminator="\n")


def check_refresh(ticker: str, old: pd.DataFrame | None, new: pd.DataFrame, start: str = START, tolerance: float = MAX_RETURN_DEVIATION) -> list[str]:
    """Vergleicht den neuen Stand eines Tickers mit dem bisherigen. Leere Liste = in Ordnung."""
    problems = []
    first = new.index[0].date().isoformat()
    if first != start:
        problems.append(f"{ticker}: Historie beginnt am {first} statt am {start}")
    if old is None or old.empty:
        return problems
    lost = old.index.difference(new.index)
    if len(lost):
        days = ", ".join(d.date().isoformat() for d in lost[:5])
        problems.append(f"{ticker}: {len(lost)} bisherige Handelstage fehlen im neuen Stand ({days})")
    if new.index[-1] < old.index[-1]:
        problems.append(f"{ticker}: neuer Stand endet am {new.index[-1].date()}, der bisherige am {old.index[-1].date()}")
    common = old.index.intersection(new.index)
    if len(common) > 1:
        old_ret = np.log(old.loc[common, "Adj Close"].astype(float)).diff().dropna()
        new_ret = np.log(new.loc[common, "Adj Close"].astype(float)).diff().dropna()
        deviation = (old_ret - new_ret).abs()
        bad = deviation[deviation > tolerance]
        if len(bad):
            worst = bad.idxmax()
            problems.append(
                f"{ticker}: {len(bad)} Tagesrenditen weichen um mehr als {tolerance:.1%} vom bisherigen Stand ab "
                f"(größte Abweichung {bad.max():.2%} am {worst.date()})"
            )
    return problems


def download_history(ticker: str, start: str = START) -> pd.DataFrame:
    """Lädt tägliche Kurse eines Tickers von Yahoo Finance (über ``yfinance``)."""
    import yfinance as yf  # nur für den Abruf nötig

    return yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=False, actions=False, raise_errors=True)


def update_directory(
    directory: str | Path,
    tickers: list[str],
    download: Callable[[str, str], pd.DataFrame] = download_history,
    now: datetime | None = None,
    start: str = START,
    dry_run: bool = False,
) -> UpdateResult:
    """Lädt alle Ticker neu, prüft sie und ersetzt die CSV-Dateien nur, wenn alles stimmt."""
    directory = Path(directory)
    cutoff = last_complete_session(now)
    problems: list[str] = []
    old: dict[str, pd.DataFrame | None] = {}
    new: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        path = directory / f"{ticker}.csv"
        old[ticker] = read_price_csv(path) if path.exists() else None
        try:
            new[ticker] = normalize_history(download(ticker, start), cutoff)
        except Exception as error:  # Netzwerk, Sperre, Formatwechsel: alles führt zum Abbruch ohne Schreiben
            problems.append(f"{ticker}: Abruf fehlgeschlagen ({type(error).__name__}: {error})")
            continue
        problems += check_refresh(ticker, old[ticker], new[ticker], start)

    last_days = {t: df.index[-1].date().isoformat() for t, df in new.items()}
    if len(set(last_days.values())) > 1:
        problems.append(f"Die Titel enden an verschiedenen Tagen: {last_days}")
    if problems:
        raise UpdateError(problems)

    previous = [df.index[-1] for df in old.values() if df is not None and not df.empty]
    previous_last = min(previous) if previous else None
    last = next(iter(new.values())).index[-1]
    reference = next(iter(new.values())).index
    new_days = [d.date().isoformat() for d in (reference if previous_last is None else reference[reference > previous_last])]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for ticker, df in new.items():
            write_price_csv(df, tmp_dir / f"{ticker}.csv")
        report = check_price_directory(tmp_dir)
        if has_blocking_issues(report):
            flagged = report[["Fehlende Werte", "Doppelte Daten", "Kurse <= 0", "Split-Verdacht"]]
            flagged = flagged[(flagged > 0).any(axis=1)]
            raise UpdateError([f"Datenprüfung meldet Fehler: {flagged.to_dict('index')}"])
        dataset = build_dataset(tmp_dir, tickers)
        if previous_last is not None and len(dataset.prices) < min(len(df) for df in old.values() if df is not None):
            raise UpdateError(["Der neue Stand hat weniger gemeinsame Handelstage als der bisherige"])

        changed = False
        for ticker in tickers:
            target = directory / f"{ticker}.csv"
            fresh = (tmp_dir / f"{ticker}.csv").read_bytes()
            if not target.exists() or target.read_bytes() != fresh:
                changed = True
                if not dry_run:
                    staged = target.with_suffix(".csv.neu")
                    staged.write_bytes(fresh)
                    os.replace(staged, target)

    notes = []
    gaps = int(report["Lücken > 5 Kalendertage"].sum())
    if gaps:
        notes.append(f"{gaps} Lücken von mehr als 5 Kalendertagen (nicht blockierend)")
    return UpdateResult(
        changed=changed,
        previous_last_day=None if previous_last is None else previous_last.date().isoformat(),
        last_day=last.date().isoformat(),
        new_days=new_days,
        notes=notes,
    )
