import numpy as np
import pandas as pd
import pytest

from portfolio_risk import load_price_csv, load_prices, log_returns, portfolio_returns, simple_returns


def write_csv(path, dates, closes):
    pd.DataFrame({"Date": dates, "Adj Close": closes}).to_csv(path, index=False)


@pytest.fixture
def price_dir(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=5).strftime("%Y-%m-%d")
    write_csv(tmp_path / "aapl.csv", dates, [100.0, 101.0, 102.0, 103.0, 104.0])
    write_csv(tmp_path / "msft.csv", dates, [200.0, 198.0, 202.0, 204.0, 206.0])
    return tmp_path


def test_load_price_csv_returns_sorted_series(tmp_path):
    path = tmp_path / "x.csv"
    write_csv(path, ["2024-01-03", "2024-01-02"], [11.0, 10.0])
    series = load_price_csv(path)
    assert list(series) == [10.0, 11.0]
    assert series.index.is_monotonic_increasing


def test_load_price_csv_rejects_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Date": ["2024-01-02"], "Close": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Adj Close"):
        load_price_csv(path)


def test_load_prices_uses_file_stem_as_ticker(price_dir):
    prices = load_prices(price_dir)
    assert list(prices.columns) == ["AAPL", "MSFT"]
    assert len(prices) == 5


def test_load_prices_drops_dates_missing_for_any_ticker(tmp_path):
    write_csv(tmp_path / "a.csv", ["2024-01-02", "2024-01-03"], [1.0, 2.0])
    write_csv(tmp_path / "b.csv", ["2024-01-03"], [5.0])
    prices = load_prices(tmp_path)
    assert list(prices.index.strftime("%Y-%m-%d")) == ["2024-01-03"]


def test_load_prices_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prices(tmp_path)


def test_simple_returns(price_dir):
    returns = simple_returns(load_prices(price_dir))
    assert len(returns) == 4
    assert returns["AAPL"].iloc[0] == pytest.approx(0.01)


def test_log_returns_match_manual_calculation(price_dir):
    prices = load_prices(price_dir)
    returns = log_returns(prices)
    assert returns["AAPL"].iloc[0] == pytest.approx(np.log(101.0 / 100.0))


def test_portfolio_returns_weighted_sum(price_dir):
    returns = simple_returns(load_prices(price_dir))
    port = portfolio_returns(returns, [0.5, 0.5])
    expected = 0.5 * returns["AAPL"] + 0.5 * returns["MSFT"]
    pd.testing.assert_series_equal(port, expected, check_names=False)


def test_portfolio_returns_rejects_bad_weights(price_dir):
    returns = simple_returns(load_prices(price_dir))
    with pytest.raises(ValueError, match="summieren"):
        portfolio_returns(returns, [0.5, 0.6])
    with pytest.raises(ValueError, match="Gewichte"):
        portfolio_returns(returns, [1.0])
