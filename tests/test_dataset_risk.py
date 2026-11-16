"""Die Risikomodelle rechnen mit der Kovarianzmatrix aus build_dataset."""

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import models, stress
from portfolio_risk.dataset import build_dataset
from portfolio_risk.models import resolve_covariance

W = [0.5, 0.3, 0.2]


@pytest.fixture
def ds(tmp_path):
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2022-01-03", periods=600).strftime("%Y-%m-%d")
    base = rng.standard_t(5, 600) * 0.01
    for name, beta in [("AAA", 1.2), ("BBB", 0.7), ("CCC", 0.2)]:
        r = beta * base + rng.normal(0, 0.01, 600)
        pd.DataFrame({"Date": dates, "Adj Close": 50 * np.exp(np.cumsum(r))}).to_csv(tmp_path / f"{name}.csv", index=False)
    return build_dataset(tmp_path, ["AAA", "BBB", "CCC"])


def test_resolve_covariance_uses_dataset_matrix(ds):
    np.testing.assert_allclose(resolve_covariance(ds.log_returns, ds.cov), ds.cov.to_numpy())
    np.testing.assert_allclose(resolve_covariance(ds.log_returns), ds.cov.to_numpy())


def test_resolve_covariance_aligns_labels(ds):
    shuffled = ds.cov.loc[["CCC", "AAA", "BBB"], ["CCC", "AAA", "BBB"]]
    np.testing.assert_allclose(resolve_covariance(ds.log_returns, shuffled), ds.cov.to_numpy())


def test_resolve_covariance_rejects_wrong_input(ds):
    with pytest.raises(ValueError, match="Form"):
        resolve_covariance(ds.log_returns, np.eye(2))
    with pytest.raises(ValueError, match="symmetrisch"):
        resolve_covariance(ds.log_returns, np.triu(np.ones((3, 3))))
    with pytest.raises(ValueError, match="ohne"):
        resolve_covariance(ds.log_returns, ds.cov.rename(index={"AAA": "XXX"}, columns={"AAA": "XXX"}))


def test_simulations_match_dataset_volatility(ds):
    sigma = ds.portfolio_volatility(W)
    normal = models.simulate_normal(ds.log_returns, W, 1, 200_000, 7, cov=ds.cov)
    t = models.simulate_student_t(ds.log_returns, W, 1, 200_000, 7, nu=5, cov=ds.cov)
    assert normal.std() == pytest.approx(sigma, rel=0.02)
    assert t.std() == pytest.approx(sigma, rel=0.03)


def test_passing_dataset_cov_is_identical_to_default(ds):
    a = models.simulate_normal(ds.log_returns, W, 3, 10_000, 3)
    b = models.simulate_normal(ds.log_returns, W, 3, 10_000, 3, cov=ds.cov)
    np.testing.assert_array_equal(a, b)
    assert models.fit_t_degrees_of_freedom(ds.log_returns, cov=ds.cov) == models.fit_t_degrees_of_freedom(ds.log_returns)
    assert stress.stressed_parametric_var(ds.log_returns, W, nu=5, cov=ds.cov) == pytest.approx(stress.stressed_parametric_var(ds.log_returns, W, nu=5))


def test_stress_responds_to_supplied_covariance(ds):
    doubled = ds.cov * 4  # doppelte Volatilität, gleiche Korrelation
    base = stress.stressed_parametric_var(ds.log_returns, W, 0.99, cov=ds.cov)
    assert stress.stressed_parametric_var(ds.log_returns, W, 0.99, cov=doubled) == pytest.approx(
        stress.stressed_parametric_var(ds.log_returns, W, 0.99, vol_factor=2.0, cov=ds.cov)
    )
    assert stress.stressed_parametric_var(ds.log_returns, W, 0.99, cov=doubled) > base
    cs = stress.conditional_shock(ds.log_returns, W, {"AAA": 1.0}, -0.3, cov=ds.cov)
    assert cs["Bewegung AAA"] == pytest.approx(-0.3)
    assert cs["Bewegung BBB"] < 0
