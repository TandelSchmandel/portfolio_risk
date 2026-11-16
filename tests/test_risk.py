import numpy as np
import pandas as pd
import pytest

from portfolio_risk.portfolios import PORTFOLIOS, TICKERS
from portfolio_risk.risk import (
    expected_shortfall,
    historical_var,
    monte_carlo_var,
    replay_return,
    uniform_shock,
    worst_window,
)


def test_historical_var_on_uniform_grid():
    returns = np.linspace(-0.10, 0.10, 201)  # 5 %-Quantil liegt bei -0.09
    assert historical_var(returns, 0.95) == pytest.approx(0.09, abs=1e-3)


def test_expected_shortfall_is_at_least_var():
    rng = np.random.default_rng(0)
    returns = rng.standard_t(df=4, size=5000) * 0.01
    assert expected_shortfall(returns, 0.99) >= historical_var(returns, 0.99)


def test_monte_carlo_var_matches_normal_theory():
    rng = np.random.default_rng(1)
    sigma = 0.01
    asset = pd.DataFrame({"X": rng.normal(0, sigma, 5000)})
    var99 = monte_carlo_var(asset, [1.0], confidence=0.99, n_sims=200_000)
    assert var99 == pytest.approx(2.326 * sigma, rel=0.05)


def test_replay_return_compounds_daily_returns():
    dates = pd.bdate_range("2022-01-03", periods=10)
    asset = pd.DataFrame({"A": [0.01] * 10, "B": [0.01] * 10}, index=dates)
    assert replay_return(asset, [0.5, 0.5], "2022-01-03", "2022-01-14") == pytest.approx(1.01**10 - 1)


def test_replay_return_rejects_empty_window():
    dates = pd.bdate_range("2022-01-03", periods=3)
    asset = pd.DataFrame({"A": [0.0] * 3}, index=dates)
    with pytest.raises(ValueError):
        replay_return(asset, [1.0], "2030-01-01", "2030-02-01")


def test_worst_window_finds_crash_period():
    dates = pd.bdate_range("2022-01-03", periods=40)
    values = np.full(40, 0.0)
    values[10:15] = -0.05  # fünf Tage mit -5 %
    asset = pd.DataFrame({"A": values}, index=dates)
    worst, end = worst_window(asset, [1.0], days=5)
    assert worst == pytest.approx((0.95**5) - 1)
    assert end == dates[14]


def test_uniform_shock_scales_with_weights():
    assert uniform_shock([0.5, 0.5], -0.2) == pytest.approx(-0.2)


def test_invalid_confidence_raises():
    with pytest.raises(ValueError):
        historical_var([0.01, -0.02], confidence=1.5)


def test_portfolio_weights_sum_to_one():
    for name, weights in PORTFOLIOS.items():
        assert len(weights) == len(TICKERS), name
        assert sum(weights) == pytest.approx(1.0), name
