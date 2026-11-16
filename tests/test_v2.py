"""Tests für den zweiten Durchlauf: Modelle, Konfidenzintervalle, Backtest, Stresstests, Datenprüfung."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from portfolio_risk import backtest, models, stress
from portfolio_risk.quality import check_price_directory, has_blocking_issues, robust_z
from portfolio_risk.risk import var_es_with_ci


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(7)
    cov = np.array([[1.0, 0.5], [0.5, 1.0]]) * 0.01**2
    x = rng.multivariate_normal([0, 0], cov, size=4000)
    return pd.DataFrame(x, columns=["A", "B"], index=pd.bdate_range("2020-01-01", periods=4000))


@pytest.fixture
def t_returns():
    rng = np.random.default_rng(11)
    nu = 4.0
    z = rng.standard_normal((20_000, 2)) @ np.linalg.cholesky([[1.0, 0.6], [0.6, 1.0]]).T
    x = z * np.sqrt(nu / rng.chisquare(nu, 20_000))[:, None] * 0.01
    return pd.DataFrame(x, columns=["A", "B"])


# --------------------------------------------------------------------------- Modelle


def test_simulations_are_reproducible_with_seed(normal_returns):
    a = models.simulate_student_t(normal_returns, [0.5, 0.5], n_sims=5000, seed=3, nu=5)
    b = models.simulate_student_t(normal_returns, [0.5, 0.5], n_sims=5000, seed=3, nu=5)
    np.testing.assert_array_equal(a, b)


def test_chunking_does_not_change_distribution(normal_returns):
    a = models.simulate_normal(normal_returns, [0.5, 0.5], n_sims=50_000, chunk=7_000)
    assert len(a) == 50_000
    assert np.std(a) == pytest.approx(np.std(normal_returns.to_numpy() @ [0.5, 0.5]), rel=0.05)


def test_fit_t_recovers_degrees_of_freedom(t_returns):
    assert models.fit_t_degrees_of_freedom(t_returns) == pytest.approx(4.0, abs=0.5)


def test_fit_t_on_normal_data_gives_large_nu(normal_returns):
    assert models.fit_t_degrees_of_freedom(normal_returns) > 15


def test_student_t_var_exceeds_normal_var_at_99(t_returns):
    w = [0.5, 0.5]
    normal = models.simulate_normal(t_returns, w, n_sims=200_000)
    t = models.simulate_student_t(t_returns, w, n_sims=200_000)
    assert -np.quantile(t, 0.01) > -np.quantile(normal, 0.01)


def test_single_asset_normal_var_matches_theory():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"X": rng.normal(0, 0.01, 5000)})
    sims = models.simulate_normal(x, [1.0], n_sims=400_000)
    sigma = x["X"].std()
    assert -np.quantile(sims, 0.01) == pytest.approx(-np.expm1(stats.norm.ppf(0.01) * sigma), rel=0.02)


def test_ten_day_normal_var_follows_sqrt_time_for_iid_normal(normal_returns):
    # Bei unabhängigen normalen Tagen ist Wurzel-Zeit korrekt; das Modell muss das reproduzieren.
    one = models.simulate_normal(normal_returns, [0.5, 0.5], horizon=1, n_sims=200_000)
    ten = models.simulate_normal(normal_returns, [0.5, 0.5], horizon=10, n_sims=200_000)
    ratio = np.log1p(np.quantile(ten, 0.01)) / np.log1p(np.quantile(one, 0.01))
    assert ratio == pytest.approx(np.sqrt(10), rel=0.03)


def test_bootstrap_one_day_draws_only_historical_values(normal_returns):
    w = np.array([0.5, 0.5])
    sims = models.simulate_block_bootstrap(normal_returns, w, horizon=1, n_sims=2000)
    centered = normal_returns - normal_returns.mean()
    allowed = np.expm1(centered.to_numpy()) @ w
    assert np.isin(np.round(sims, 12), np.round(allowed, 12)).all()


def test_block_bootstrap_uses_consecutive_days():
    x = pd.DataFrame({"A": np.arange(20, dtype=float) * 1e-3})
    sims = models.simulate_block_bootstrap(x, [1.0], horizon=4, block=4, n_sims=500)
    centered = x["A"].to_numpy() - x["A"].mean()
    # Jede Summe muss die Summe von vier aufeinanderfolgenden Tagen sein (zirkulär).
    sums = {round(np.expm1(sum(centered[(s + k) % 20] for k in range(4))), 10) for s in range(20)}
    assert set(np.round(sims, 10)) <= sums


def test_ewma_volatility_has_no_lookahead():
    x = np.zeros(50)
    x[40] = 0.1
    sigma, _ = models.ewma_volatility(x, warmup=10)
    assert sigma[40] == sigma[39] * np.sqrt(0.94)  # Schock an Tag 40 wirkt erst ab Tag 41
    assert sigma[41] > sigma[40]


def test_fhs_scales_with_current_volatility():
    rng = np.random.default_rng(5)
    calm_end = np.concatenate([rng.normal(0, 0.02, 500), rng.normal(0, 0.005, 100)])
    wild_end = np.concatenate([rng.normal(0, 0.005, 500), rng.normal(0, 0.02, 100)])
    var_calm = -np.quantile(models.simulate_filtered_historical(calm_end, [1.0], n_sims=50_000), 0.01)
    var_wild = -np.quantile(models.simulate_filtered_historical(wild_end, [1.0], n_sims=50_000), 0.01)
    assert var_wild > 2 * var_calm


def test_historical_horizon_returns_compound_per_asset():
    lr = pd.DataFrame({"A": [np.log(1.1)] * 3, "B": [0.0] * 3})
    out = models.historical_horizon_returns(lr, [0.5, 0.5], horizon=3)
    assert len(out) == 1
    assert out.iloc[0] == pytest.approx(0.5 * (1.1**3 - 1))


def test_bad_weights_raise(normal_returns):
    with pytest.raises(ValueError, match="summieren"):
        models.simulate_normal(normal_returns, [0.5, 0.6], n_sims=10)


# --------------------------------------------------------------------------- Konfidenzintervall


def test_var_ci_contains_true_quantile_and_shrinks_with_n():
    rng = np.random.default_rng(2)
    true_var = -stats.norm.ppf(0.01)
    small = var_es_with_ci(rng.standard_normal(5_000), 0.99)
    large = var_es_with_ci(rng.standard_normal(500_000), 0.99)
    for res in (small, large):
        assert res["VaR KI unten"] <= true_var <= res["VaR KI oben"]
    assert (large["VaR KI oben"] - large["VaR KI unten"]) < (small["VaR KI oben"] - small["VaR KI unten"]) / 5


def test_var_ci_wider_with_effective_n():
    r = np.random.default_rng(3).standard_normal(2000)
    full = var_es_with_ci(r, 0.99)
    reduced = var_es_with_ci(r, 0.99, effective_n=200)
    assert reduced["VaR KI oben"] - reduced["VaR KI unten"] > full["VaR KI oben"] - full["VaR KI unten"]


def test_es_at_least_var():
    r = np.random.default_rng(4).standard_t(4, 10_000)
    res = var_es_with_ci(r, 0.99)
    assert res["ES"] >= res["VaR"]


# --------------------------------------------------------------------------- Backtest


def test_kupiec_accepts_exact_rate_and_rejects_far_off():
    _, p_exact = backtest.kupiec_pof(10, 1000, 0.99)
    _, p_bad = backtest.kupiec_pof(30, 1000, 0.99)
    assert p_exact == pytest.approx(1.0)
    assert p_bad < 0.001


def test_kupiec_handles_zero_exceptions():
    lr, p = backtest.kupiec_pof(0, 500, 0.99)
    assert lr == pytest.approx(-2 * 500 * np.log(0.99))
    assert 0 < p < 0.05


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, dtype=int)
    clustered[100:110] = 1
    spread = np.zeros(1000, dtype=int)
    spread[::100] = 1
    assert backtest.christoffersen_independence(clustered)[1] < 0.001
    assert backtest.christoffersen_independence(spread)[1] > 0.05


def test_traffic_light():
    assert backtest.basel_traffic_light(4) == "grün"
    assert backtest.basel_traffic_light(5) == "gelb"
    assert backtest.basel_traffic_light(10) == "rot"


def test_rolling_forecasts_are_out_of_sample():
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0, 0.01, 400), index=pd.bdate_range("2020-01-01", periods=400))
    f1 = backtest.rolling_var_forecasts(r, 0.99, window=250)
    r2 = r.copy()
    r2.iloc[300] = -0.5  # Ein Crash an Tag 300 darf die Prognose für Tag 300 nicht verändern
    f2 = backtest.rolling_var_forecasts(r2, 0.99, window=250)
    day = r.index[300]
    pd.testing.assert_series_equal(f1.drop(columns="Verlust").loc[day], f2.drop(columns="Verlust").loc[day])
    assert f2.loc[day, "Verlust"] == pytest.approx(0.5)


def test_evaluate_on_correct_model_is_not_rejected():
    rng = np.random.default_rng(10)
    r = pd.Series(rng.normal(0, 0.01, 1500), index=pd.bdate_range("2015-01-01", periods=1500))
    res = backtest.evaluate(backtest.rolling_var_forecasts(r, 0.99), 0.99)
    assert res.loc["Normal", "Kupiec p"] > 0.05


# --------------------------------------------------------------------------- Stresstests


@pytest.fixture
def two_assets():
    idx = pd.bdate_range("2022-01-03", periods=6)
    return pd.DataFrame({"A": np.log([1.0, 0.9, 0.9, 1.0, 1.0, 1.0]), "B": [0.0] * 6}, index=idx)


def test_scenario_return_is_buy_and_hold(two_assets):
    # A: 1.0 * 0.9 * 0.9 = 0.81 -> -19 %, Gewicht 50 %
    assert stress.scenario_return(two_assets, [0.5, 0.5], "2022-01-04", "2022-01-05") == pytest.approx(0.5 * -0.19)


def test_scenario_outside_data_raises(two_assets):
    with pytest.raises(ValueError):
        stress.scenario_return(two_assets, [0.5, 0.5], "2030-01-01", "2030-01-02")


def test_worst_window_dates(two_assets):
    ww = stress.worst_window(two_assets, [1.0, 0.0], days=2)
    assert ww["Rendite"] == pytest.approx(0.81 - 1)
    assert (ww["Start"], ww["Ende"]) == ("2022-01-04", "2022-01-05")


def test_max_drawdown_and_recovery(two_assets):
    dd = stress.max_drawdown(two_assets, [1.0, 0.0])
    assert dd["Drawdown"] == pytest.approx(0.81 - 1)
    assert dd["Tief"] == "2022-01-05"


def test_conditional_shock_moves_correlated_assets(normal_returns):
    cs = stress.conditional_shock(normal_returns, [0.5, 0.5], {"A": 1.0}, -0.30)
    assert cs["Bewegung A"] == pytest.approx(-0.30)
    # Korrelation 0.5, gleiche Vola: B fällt im Mittel etwa halb so stark (in Log-Renditen)
    assert np.log1p(cs["Bewegung B"]) == pytest.approx(0.5 * np.log(0.7), rel=0.1)


def test_correlation_stress_increases_var(normal_returns):
    base = stress.stressed_parametric_var(normal_returns, [0.5, 0.5], 0.99)
    stressed = stress.stressed_parametric_var(normal_returns, [0.5, 0.5], 0.99, corr=0.95)
    assert stressed > base


def test_sector_shock():
    assert stress.sector_shock(["A", "B"], [0.25, 0.75], {"A": -0.4}, default=-0.1) == pytest.approx(-0.175)


def test_historical_scenarios_have_valid_dates():
    for sc in stress.HISTORICAL_SCENARIOS:
        assert pd.Timestamp(sc.start) <= pd.Timestamp(sc.end), sc.name


# --------------------------------------------------------------------------- Datenprüfung


def test_quality_flags_split_and_outlier(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=60).strftime("%Y-%m-%d")
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    prices[30:] /= 10  # nicht bereinigter 10:1-Split
    pd.DataFrame({"Date": dates, "Close": prices, "Adj Close": prices}).to_csv(tmp_path / "x.csv", index=False)
    report = check_price_directory(tmp_path)
    assert report.loc["X", "Split-Verdacht"] == 1
    assert report.loc["X", "Ausreißer |robust z| > 8"] >= 1
    assert has_blocking_issues(report)


def test_quality_clean_data_passes(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=60).strftime("%Y-%m-%d")
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 60)))
    pd.DataFrame({"Date": dates, "Adj Close": prices}).to_csv(tmp_path / "y.csv", index=False)
    assert not has_blocking_issues(check_price_directory(tmp_path))


def test_robust_z_ignores_outlier_in_scale():
    s = pd.Series([0.0, 0.01, -0.01, 0.005, -0.005, 1.0])
    assert robust_z(s).iloc[-1] > 50
