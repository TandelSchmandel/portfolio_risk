"""Prüft den JSON-Export für das Dashboard gegen die Tabellen aus Schritt 2."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from portfolio_risk import stress
from portfolio_risk.data import portfolio_returns
from portfolio_risk.dataset import build_dataset
from portfolio_risk.portfolios import PORTFOLIOS, TICKERS

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("export_dashboard", ROOT / "scripts" / "export_dashboard.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@pytest.fixture(scope="module")
def ds():
    return build_dataset(ROOT / "data" / "raw", TICKERS)


def test_var_records_match_summary():
    records = export.var_records()
    summary = export.summary_table()
    for name in PORTFOLIOS:
        rec = next(
            r for r in records if r["portfolio"] == name and r["modell"] == "student_t" and r["horizont"] == 1 and r["konfidenz"] == 0.99
        )
        assert rec["var"] == pytest.approx(summary[name]["VaR 99 % 1T Student-t"], abs=1e-4)
        assert rec["var_ki"][0] <= rec["var"] <= rec["var_ki"][1]


def test_every_stress_scenario_has_all_portfolios():
    for rec in export.stress_records():
        assert set(rec["werte"]) == set(PORTFOLIOS)


def test_drawdown_matches_stress_module(ds):
    for name, w in PORTFOLIOS.items():
        r = portfolio_returns(ds.simple_returns, w)
        series = export.portfolio_series(ds.log_returns, w, r, with_backtest=False)
        assert len(series["wert"]) == len(ds.log_returns)
        assert min(series["drawdown"]) == pytest.approx(stress.max_drawdown(ds.log_returns, w)["Drawdown"], abs=1e-6)
        assert "backtest_99" not in series


def test_clean_produces_valid_json():
    payload = export.clean({"a": np.float64("nan"), "b": np.int64(3), "c": [np.float32(0.1234567891)], "d": np.bool_(True)})
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text) == {"a": None, "b": 3, "c": [round(float(np.float32(0.1234567891)), 6)], "d": True}
