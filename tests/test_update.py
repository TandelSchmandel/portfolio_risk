"""Fortschreiben der Kursdaten: Format, Prüfungen und Alles-oder-nichts beim Schreiben (ohne Netzwerk)."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.dataset import build_dataset
from portfolio_risk.update import (
    COLUMNS,
    UpdateError,
    check_refresh,
    last_complete_session,
    normalize_history,
    read_price_csv,
    update_directory,
    write_price_csv,
)

TICKERS = ["AAA", "BBB", "CCC"]
START = "2024-01-02"
NOW = datetime(2024, 12, 2, 5, 23, tzinfo=timezone.utc)  # Montagmorgen UTC, in New York noch Sonntag


def yahoo_frame(days: int, seed: int, start: str = START, dividend_on: int | None = None) -> pd.DataFrame:
    """Tabelle, wie sie ``yfinance.Ticker.history(auto_adjust=False)`` liefert: Zeitzone New York, sechs Spalten."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=days, tz="America/New_York", name="Date")
    close = 80 * np.exp(np.cumsum(rng.normal(0, 0.02, days)))
    adj = close.copy()
    if dividend_on is not None:
        adj[:dividend_on] *= 0.995  # Yahoo rechnet alle Kurse vor dem Ex-Tag herunter
    return pd.DataFrame(
        {"Open": close * 0.99, "High": close * 1.01, "Low": close * 0.98, "Close": close, "Adj Close": adj,
         "Volume": rng.integers(1_000_000, 9_000_000, days).astype(float)},
        index=index,
    )


def downloader(frames: dict[str, pd.DataFrame]):
    calls = []

    def download(ticker, start):
        calls.append((ticker, start))
        frame = frames[ticker]
        if isinstance(frame, Exception):
            raise frame
        return frame

    download.calls = calls
    return download


@pytest.fixture
def data_dir(tmp_path):
    """Bisheriger Stand: 200 Tage je Ticker, mit 2 Nachkommastellen wie die ersten Dateien des Projekts."""
    for i, ticker in enumerate(TICKERS):
        old = normalize_history(yahoo_frame(200, seed=i), date(2030, 1, 1))
        old[["Open", "High", "Low", "Close", "Adj Close"]] = old[["Open", "High", "Low", "Close", "Adj Close"]].round(2)
        old.index = old.index.strftime("%Y-%m-%d")
        old.to_csv(tmp_path / f"{ticker}.csv", index_label="Date", float_format="%.2f")
    return tmp_path


def fresh(days: int, **kwargs) -> dict[str, pd.DataFrame]:
    return {ticker: yahoo_frame(days, seed=i, **kwargs) for i, ticker in enumerate(TICKERS)}


# --------------------------------------------------------------------------- Zeit und Format


@pytest.mark.parametrize(
    "now, expected",
    [
        (datetime(2026, 11, 17, 5, 23, tzinfo=timezone.utc), date(2026, 11, 16)),  # 00:23 in New York: Montag ist fertig
        (datetime(2026, 11, 16, 20, 0, tzinfo=timezone.utc), date(2026, 11, 15)),  # 15 Uhr in New York: Handel läuft noch
        (datetime(2026, 11, 16, 22, 0, tzinfo=timezone.utc), date(2026, 11, 16)),  # 17 Uhr in New York: abgeschlossen
        (datetime(2026, 7, 6, 20, 30, tzinfo=timezone.utc), date(2026, 7, 5)),  # Sommerzeit: 16:30 Uhr, noch im Puffer
    ],
)
def test_last_complete_session(now, expected):
    assert last_complete_session(now) == expected


def test_normalize_history_matches_project_format():
    raw = yahoo_frame(30, seed=5)
    raw.iloc[10, raw.columns.get_loc("Close")] = np.nan  # Yahoo liefert vereinzelt leere Zeilen
    raw.iloc[10, raw.columns.get_loc("Adj Close")] = np.nan
    out = normalize_history(raw, cutoff=raw.index[-2].date())
    assert list(out.columns) == COLUMNS and out.index.name == "Date" and out.index.tz is None
    assert len(out) == 28  # letzter Tag liegt nach dem Stichtag, eine Zeile war leer
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert out["Volume"].dtype == "int64"
    assert (out[["Open", "High", "Low", "Close", "Adj Close"]].round(4) == out[["Open", "High", "Low", "Close", "Adj Close"]]).all().all()


def test_normalize_history_accepts_download_layout_and_rejects_bad_input():
    raw = yahoo_frame(5, seed=1)
    multi = raw.copy()
    multi.columns = pd.MultiIndex.from_product([raw.columns, ["AAA"]])
    pd.testing.assert_frame_equal(normalize_history(multi, date(2030, 1, 1)), normalize_history(raw, date(2030, 1, 1)))
    with pytest.raises(ValueError):
        normalize_history(raw.drop(columns="Adj Close"), date(2030, 1, 1))
    with pytest.raises(ValueError):
        normalize_history(raw.iloc[:0], date(2030, 1, 1))
    with pytest.raises(ValueError):
        normalize_history(raw, date(2000, 1, 1))  # alles nach dem Stichtag
    negative = raw.copy()
    negative.iloc[2, 0] = -1.0
    with pytest.raises(ValueError):
        normalize_history(negative, date(2030, 1, 1))


def test_csv_round_trip(tmp_path):
    df = normalize_history(yahoo_frame(20, seed=2), date(2030, 1, 1))
    write_price_csv(df, tmp_path / "AAA.csv")
    header, first = (tmp_path / "AAA.csv").read_text().splitlines()[:2]
    assert header == "Date,Open,High,Low,Close,Adj Close,Volume"
    assert first.startswith("2024-01-02,") and len(first.split(",")[1].split(".")[1]) == 4
    pd.testing.assert_frame_equal(read_price_csv(tmp_path / "AAA.csv"), df, check_freq=False)


# --------------------------------------------------------------------------- Vergleich mit dem bisherigen Stand


def test_check_refresh_accepts_extension_rounding_and_dividend():
    old = normalize_history(yahoo_frame(200, seed=0), date(2030, 1, 1)).round(2)
    new = normalize_history(yahoo_frame(205, seed=0, dividend_on=150), date(2030, 1, 1))
    assert check_refresh("AAA", old, new, START) == []
    assert check_refresh("AAA", None, new, START) == []


def test_check_refresh_reports_each_problem():
    old = normalize_history(yahoo_frame(200, seed=0), date(2030, 1, 1))
    assert "beginnt am" in check_refresh("AAA", old, old.iloc[1:], START)[0]
    assert any("fehlen" in p for p in check_refresh("AAA", old, old.drop(old.index[50]), START))
    assert any("endet am" in p for p in check_refresh("AAA", old, old.iloc[:-3], START))
    wrong = old.copy()
    wrong.iloc[100:, wrong.columns.get_loc("Adj Close")] *= 0.5  # nicht bereinigter Split
    problems = check_refresh("AAA", old, wrong, START)
    assert len(problems) == 1 and "weichen" in problems[0] and old.index[100].date().isoformat() in problems[0]


# --------------------------------------------------------------------------- Ganzer Ablauf


def test_update_appends_new_days_and_keeps_dataset_loadable(data_dir):
    before = build_dataset(data_dir, TICKERS)
    download = downloader(fresh(205))
    result = update_directory(data_dir, TICKERS, download, now=NOW, start=START)
    after = build_dataset(data_dir, TICKERS)
    assert result.changed and len(result.new_days) == 5
    assert result.previous_last_day == before.prices.index[-1].date().isoformat()
    assert result.last_day == after.prices.index[-1].date().isoformat() == result.new_days[-1]
    assert len(after.prices) == len(before.prices) + 5
    assert download.calls == [(t, START) for t in TICKERS]
    # Die gemeinsamen Tage liefern dieselben Renditen (bis auf die Rundung des alten Stands)
    common = before.log_returns.index
    assert np.abs(after.log_returns.loc[common] - before.log_returns).to_numpy().max() < 1e-3


def test_second_run_without_new_data_changes_nothing(data_dir):
    update_directory(data_dir, TICKERS, downloader(fresh(205)), now=NOW, start=START)
    stamp = {t: (data_dir / f"{t}.csv").stat().st_mtime_ns for t in TICKERS}
    result = update_directory(data_dir, TICKERS, downloader(fresh(205)), now=NOW, start=START)
    assert not result.changed and result.new_days == []
    assert stamp == {t: (data_dir / f"{t}.csv").stat().st_mtime_ns for t in TICKERS}


def test_running_session_is_not_stored(data_dir):
    frames = fresh(205)
    last_day = frames["AAA"].index[-1]
    during_trading = last_day.replace(hour=11)  # 11 Uhr in New York am letzten gelieferten Tag
    result = update_directory(data_dir, TICKERS, downloader(frames), now=during_trading.to_pydatetime(), start=START)
    assert len(result.new_days) == 4 and result.last_day < last_day.date().isoformat()


def test_dry_run_writes_nothing(data_dir):
    before = {t: (data_dir / f"{t}.csv").read_bytes() for t in TICKERS}
    result = update_directory(data_dir, TICKERS, downloader(fresh(205)), now=NOW, start=START, dry_run=True)
    assert result.changed and len(result.new_days) == 5
    assert before == {t: (data_dir / f"{t}.csv").read_bytes() for t in TICKERS}
    assert sorted(p.name for p in data_dir.iterdir()) == [f"{t}.csv" for t in TICKERS]


@pytest.mark.parametrize("case", ["download_fails", "ticker_lags", "history_changed", "day_missing", "split_error"])
def test_any_problem_leaves_all_files_untouched(data_dir, case):
    frames = fresh(205)
    if case == "download_fails":
        frames["BBB"] = ConnectionError("429 Too Many Requests")
    elif case == "ticker_lags":
        frames["CCC"] = frames["CCC"].iloc[:-1]
    elif case == "history_changed":
        frames["AAA"] = frames["AAA"].copy()
        frames["AAA"].iloc[:60, frames["AAA"].columns.get_loc("Adj Close")] *= 1.2
    elif case == "day_missing":
        frames["BBB"] = frames["BBB"].drop(frames["BBB"].index[40])
    elif case == "split_error":
        frames["CCC"] = frames["CCC"].copy()
        frames["CCC"].iloc[202:, :5] *= 0.25  # neuer Tag mit unbereinigtem 4:1-Split
    before = {t: (data_dir / f"{t}.csv").read_bytes() for t in TICKERS}
    with pytest.raises(UpdateError) as error:
        update_directory(data_dir, TICKERS, downloader(frames), now=NOW, start=START)
    assert error.value.problems
    assert before == {t: (data_dir / f"{t}.csv").read_bytes() for t in TICKERS}
    assert sorted(p.name for p in data_dir.iterdir()) == [f"{t}.csv" for t in TICKERS]


def test_first_download_into_empty_directory(tmp_path):
    result = update_directory(tmp_path, TICKERS, downloader(fresh(120)), now=NOW, start=START)
    assert result.changed and result.previous_last_day is None and len(result.new_days) == 120
    assert len(build_dataset(tmp_path, TICKERS).prices) == 120


def test_cli_reports_and_sets_github_outputs(data_dir, tmp_path_factory, monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    from portfolio_risk import update as update_module

    script = Path(__file__).resolve().parent.parent / "scripts" / "update_prices.py"
    spec = importlib.util.spec_from_file_location("update_prices", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    output = tmp_path_factory.mktemp("gh") / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(cli, "TICKERS", TICKERS)
    frames = fresh(203)
    last_day = frames["AAA"].index[-1].date().isoformat()

    def fake_update(directory, tickers, dry_run=False):
        return update_module.update_directory(directory, tickers, downloader(frames), now=NOW, start=START, dry_run=dry_run)

    monkeypatch.setattr(cli, "update_directory", fake_update)
    assert cli.main(["--data-dir", str(data_dir)]) == 0
    assert "3 neue Handelstage" in capsys.readouterr().out
    assert output.read_text() == f"changed=true\nlast_day={last_day}\n"

    frames["BBB"] = TimeoutError("keine Antwort")
    assert cli.main(["--data-dir", str(data_dir)]) == 1
    assert "BBB: Abruf fehlgeschlagen" in capsys.readouterr().err
