import numpy as np
import pandas as pd
import pytest

from portfolio_risk.dataset import (
    TRADING_DAYS,
    build_dataset,
    is_positive_definite,
    sample_covariance,
    shrunk_covariance,
)
from portfolio_risk.portfolios import PORTFOLIOS, TICKERS


@pytest.fixture
def price_dir(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=400).strftime("%Y-%m-%d")
    base = rng.normal(0, 0.01, 400)
    for name, beta in [("AAA", 1.0), ("BBB", 0.5), ("CCC", 0.0)]:
        r = beta * base + rng.normal(0, 0.01, 400)
        pd.DataFrame({"Date": dates, "Adj Close": 100 * np.exp(np.cumsum(r))}).to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path


def test_build_dataset_shapes_and_order(price_dir):
    ds = build_dataset(price_dir, ["CCC", "AAA", "BBB"])
    assert ds.tickers == ["CCC", "AAA", "BBB"]
    assert len(ds.log_returns) == len(ds.prices) - 1
    assert list(ds.cov.columns) == list(ds.cov.index) == ["CCC", "AAA", "BBB"]


def test_log_returns_match_prices(price_dir):
    ds = build_dataset(price_dir)
    manual = np.log(ds.prices["AAA"].iloc[1] / ds.prices["AAA"].iloc[0])
    assert ds.log_returns["AAA"].iloc[0] == pytest.approx(manual)
    assert ds.simple_returns["AAA"].iloc[0] == pytest.approx(np.expm1(manual))


def test_covariance_is_symmetric_psd_and_matches_numpy(price_dir):
    ds = build_dataset(price_dir)
    cov = ds.cov.to_numpy()
    np.testing.assert_allclose(cov, cov.T)
    np.testing.assert_allclose(cov, np.cov(ds.log_returns.to_numpy(), rowvar=False))
    assert is_positive_definite(ds.cov)


def test_covariance_diagonal_is_variance_and_corr_in_range(price_dir):
    ds = build_dataset(price_dir)
    np.testing.assert_allclose(np.diag(ds.cov), ds.vol_daily**2)
    assert (ds.corr.abs() <= 1 + 1e-12).all().all()
    assert ds.corr.loc["AAA", "BBB"] > ds.corr.loc["AAA", "CCC"]


def test_annualisation(price_dir):
    ds = build_dataset(price_dir)
    np.testing.assert_allclose(ds.cov_annual.to_numpy(), ds.cov.to_numpy() * TRADING_DAYS)
    np.testing.assert_allclose(ds.vol_annual, ds.vol_daily * np.sqrt(TRADING_DAYS))


def test_portfolio_volatility_matches_return_series(price_dir):
    ds = build_dataset(price_dir)
    w = np.array([0.5, 0.3, 0.2])
    direct = (ds.log_returns.to_numpy() @ w).std(ddof=1)
    assert ds.portfolio_volatility(w) == pytest.approx(direct)
    with pytest.raises(ValueError):
        ds.portfolio_volatility([0.5, 0.5])


def test_unknown_ticker_raises(price_dir):
    with pytest.raises(ValueError, match="ZZZ"):
        build_dataset(price_dir, ["AAA", "ZZZ"])


def test_nonpositive_price_raises(tmp_path):
    dates = pd.bdate_range("2022-01-03", periods=10).strftime("%Y-%m-%d")
    pd.DataFrame({"Date": dates, "Adj Close": [1.0] * 9 + [0.0]}).to_csv(tmp_path / "X.csv", index=False)
    with pytest.raises(ValueError, match="positiv"):
        build_dataset(tmp_path)


def test_sample_covariance_rejects_nan_and_short_samples():
    with pytest.raises(ValueError, match="fehlende"):
        sample_covariance(pd.DataFrame({"A": [0.1, np.nan, 0.2], "B": [0.0, 0.1, 0.2]}))
    with pytest.raises(ValueError, match="Zu wenige"):
        sample_covariance(pd.DataFrame({"A": [0.1, 0.2], "B": [0.0, 0.1]}))


def test_shrinkage_in_unit_interval_and_psd(price_dir):
    ds = build_dataset(price_dir)
    shrunk, intensity = shrunk_covariance(ds.log_returns)
    assert 0.0 <= intensity <= 1.0
    assert is_positive_definite(shrunk)
    # Spur bleibt (nahezu) erhalten, Schrumpfung verändert nur die Struktur
    assert np.trace(shrunk.to_numpy()) == pytest.approx(np.trace(ds.cov.to_numpy()), rel=0.01)


def test_shrinkage_is_larger_with_fewer_observations():
    rng = np.random.default_rng(3)
    # Struktur mit Faktor und unterschiedlichen Varianzen, damit die Zielmatrix (Einheitsmatrix) nicht exakt passt
    factor = rng.normal(0, 0.01, (2000, 1))
    loadings = np.array([[0.2, 0.5, 1.0, 1.5, 2.0]])
    x = pd.DataFrame(factor @ loadings + rng.normal(0, 0.005, (2000, 5)), columns=list("ABCDE"))
    _, small = shrunk_covariance(x.iloc[:30])
    _, large = shrunk_covariance(x)
    assert 0 <= large < small <= 1


def test_is_positive_definite_detects_singular():
    assert not is_positive_definite(np.array([[1.0, 1.0], [1.0, 1.0]]))


def test_real_data_if_present():
    from pathlib import Path

    raw = Path(__file__).resolve().parent.parent / "data" / "raw"
    if not raw.exists():
        pytest.skip("keine Rohdaten")
    ds = build_dataset(raw, TICKERS)
    summary = ds.summary()
    assert summary["Positiv definit"]
    assert summary["Tagesrenditen"] == 1253
    assert summary["Konditionszahl"] < 100
    for weights in PORTFOLIOS.values():
        assert 0.01 < ds.portfolio_volatility(weights) < 0.05
