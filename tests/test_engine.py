"""Rechenkern des Dashboards: gleiche Ergebnisse wie die Funktionen aus Schritt 2, nur für beliebige Gewichte."""

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import models, stress
from portfolio_risk.dataset import build_dataset
from portfolio_risk.engine import MAIN_MODEL, RISK_MODELS, EngineConfig, RiskEngine
from portfolio_risk.portfolios import GROUPS, NAMES, PORTFOLIOS, TICKERS
from portfolio_risk.risk import var_es_with_ci

CONFIG = EngineConfig(n_sims=6_000, chunk=2_500, max_horizon=12)


def write_prices(directory, days=420, seed=3):
    """Zehn künstliche Kursreihen mit gemeinsamem Marktfaktor und dicken Rändern."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=days).strftime("%Y-%m-%d")
    market = rng.standard_t(5, days) * 0.012
    for i, ticker in enumerate(TICKERS):
        r = (0.6 + 0.1 * i) * market + rng.normal(0, 0.012, days)
        pd.DataFrame({"Date": dates, "Adj Close": 40 * np.exp(np.cumsum(r))}).to_csv(directory / f"{ticker}.csv", index=False)


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    directory = tmp_path_factory.mktemp("kurse")
    write_prices(directory)
    return build_dataset(directory, TICKERS)


@pytest.fixture(scope="module")
def engine(ds):
    return RiskEngine(ds, CONFIG)


def random_weights(seed):
    return np.random.default_rng(seed).dirichlet(np.ones(len(TICKERS)))


# --------------------------------------------------------------------------- Szenarien


@pytest.mark.parametrize("model", models.SCENARIO_MODELS)
@pytest.mark.parametrize("horizon", [1, 7])
def test_scenario_returns_equal_simulation_for_any_weights(ds, model, horizon):
    simulate = {
        "normal": lambda w: models.simulate_normal(ds.log_returns, w, horizon, 5_000, 9, chunk=2_000, cov=ds.cov),
        "student_t": lambda w: models.simulate_student_t(ds.log_returns, w, horizon, 5_000, 9, nu=5.0, chunk=2_000, cov=ds.cov),
        "bootstrap": lambda w: models.simulate_block_bootstrap(ds.log_returns, w, horizon, 5_000, 9, chunk=2_000),
        "fhs": lambda w: models.simulate_filtered_historical(ds.log_returns, w, horizon, 5_000, 9, chunk=2_000),
    }[model]
    kwargs = {"cov": ds.cov} if model in ("normal", "student_t") else {}
    if model == "student_t":
        kwargs["nu"] = 5.0
    matrix = models.scenario_returns(model, ds.log_returns, horizon, 5_000, 9, chunk=2_000, **kwargs)
    assert matrix.shape == (5_000, len(TICKERS))
    for seed in (0, 1):
        w = random_weights(seed)
        np.testing.assert_array_equal(matrix @ w, simulate(w))


def test_scenario_returns_rejects_unknown_model_and_horizon(ds):
    with pytest.raises(ValueError):
        models.scenario_returns("garch", ds.log_returns)
    with pytest.raises(ValueError):
        models.scenario_returns("normal", ds.log_returns, horizon=0)


@pytest.mark.parametrize("model", models.PATH_MODELS)
def test_scenario_paths_contain_every_shorter_horizon(ds, model):
    kwargs = {"cov": ds.cov} if model in ("normal", "student_t") else {}
    if model == "student_t":
        kwargs["nu"] = 5.0
    paths = models.scenario_paths(model, ds.log_returns, 6, 3_000, 9, dtype=np.float64, **kwargs)
    assert paths.shape == (6, 3_000, len(TICKERS))
    for horizon in (1, 4, 6):
        single = models.scenario_returns(model, ds.log_returns, horizon, 3_000, 9, chunk=3_000, **kwargs)
        np.testing.assert_array_equal(paths[horizon - 1], single)
    assert models.scenario_paths(model, ds.log_returns, 2, 100, 9, **kwargs).dtype == np.float32


def test_scenario_paths_reject_bootstrap_and_bad_horizon(ds):
    with pytest.raises(ValueError):
        models.scenario_paths("bootstrap", ds.log_returns, 5)
    with pytest.raises(ValueError):
        models.scenario_paths("normal", ds.log_returns, 0)


def test_engine_scenarios_are_cached_and_evicted(ds):
    small = RiskEngine(ds, EngineConfig(n_sims=500, chunk=500, max_horizon=4, default_horizon=2, cache_models=1, cache_matrices=2))
    first = small.scenarios("normal", 1)
    assert first.shape == (500, len(TICKERS)) and first.dtype == np.float32 and not first.flags.writeable
    assert small.scenarios("normal", 3).base is first.base  # alle Horizonte eines Pfadmodells liegen in einem Block
    small.scenarios("fhs", 1)  # verdrängt das andere Pfadmodell
    again = small.scenarios("normal", 1)
    assert again.base is not first.base
    np.testing.assert_array_equal(again, first)

    boot = small.scenarios("bootstrap", 1)
    assert small.scenarios("bootstrap", 1) is boot
    small.scenarios("bootstrap", 2)
    small.scenarios("bootstrap", 3)  # verdrängt den ältesten Eintrag
    assert small.scenarios("bootstrap", 1) is not boot


def test_concurrent_requests_simulate_only_once(ds, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    small = RiskEngine(ds, EngineConfig(n_sims=2_000, max_horizon=5, default_horizon=2))
    calls = []
    original = models.scenario_paths
    monkeypatch.setattr(models, "scenario_paths", lambda *a, **k: (calls.append(a[0]), original(*a, **k))[1])
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda i: small.risk(random_weights(i), 0.95, 1 + i % 5)["student_t"]["var"], range(32)))
    assert calls == ["student_t"] and all(r > 0 for r in results)


# --------------------------------------------------------------------------- VaR und ES


@pytest.mark.parametrize("horizon", [1, 10])
def test_risk_matches_step_2_functions(engine, ds, horizon):
    w = random_weights(4)
    cfg = engine.config
    result = engine.risk(w, 0.99, horizon, RISK_MODELS)
    expected = {
        "normal": models.simulate_normal(ds.log_returns, w, horizon, cfg.n_sims, cfg.seed, chunk=cfg.n_sims, cov=ds.cov),
        "student_t": models.simulate_student_t(ds.log_returns, w, horizon, cfg.n_sims, cfg.seed, nu=engine.nu, chunk=cfg.n_sims, cov=ds.cov),
        "bootstrap": models.simulate_block_bootstrap(ds.log_returns, w, horizon, cfg.n_sims, cfg.seed),
        "fhs": models.simulate_filtered_historical(ds.log_returns, w, horizon, cfg.n_sims, cfg.seed, chunk=cfg.n_sims),
    }
    for name, samples in expected.items():
        ref = var_es_with_ci(samples, 0.99)
        assert result[name]["var"] == pytest.approx(ref["VaR"], abs=2e-6)  # Szenarien liegen als float32 vor
        assert result[name]["es"] == pytest.approx(ref["ES"], abs=2e-6)
        assert result[name]["var_ci"] == pytest.approx([ref["VaR KI unten"], ref["VaR KI oben"]], abs=2e-6)
        assert result[name]["n"] == cfg.n_sims
    hist = models.historical_horizon_returns(ds.log_returns, w, horizon).to_numpy()
    ref = var_es_with_ci(hist, 0.99, effective_n=len(hist) / horizon)
    assert result["historical"]["var"] == pytest.approx(ref["VaR"], abs=1e-12)
    assert result["historical"]["es"] == pytest.approx(ref["ES"], abs=1e-12)


def test_risk_grows_with_confidence_and_horizon(engine):
    w = engine.weights_vector(dict(zip(TICKERS, PORTFOLIOS["A_Hauptportfolio"])))
    var = lambda c, h: engine.risk(w, c, h)[MAIN_MODEL]["var"]  # noqa: E731
    assert var(0.95, 1) < var(0.99, 1) < var(0.99, 10)
    one = engine.risk(w, 0.99, 5)[MAIN_MODEL]
    assert one["var_ci"][0] <= one["var"] <= one["var_ci"][1] <= one["es"] * 1.5
    assert one["es"] >= one["var"]


def test_default_model_is_main_model(engine):
    assert list(engine.risk(random_weights(1), 0.95, 1)) == [MAIN_MODEL]


# --------------------------------------------------------------------------- Gewichte und Grenzen


def test_weights_are_normalised_and_ordered(engine):
    w = engine.weights_vector({"MSFT": 30, "NVDA": 50, "AMZN": 20})
    assert w.sum() == pytest.approx(1.0)
    assert dict(zip(TICKERS, w))["NVDA"] == pytest.approx(0.5)
    assert dict(zip(TICKERS, w))["AVGO"] == 0.0
    np.testing.assert_allclose(engine.weights_vector([2.0] * len(TICKERS)), 0.1)


@pytest.mark.parametrize(
    "weights",
    [{"XXX": 1.0}, {"NVDA": -0.1, "MSFT": 1.1}, {"NVDA": 0.0}, {"NVDA": float("nan")}, {"NVDA": float("inf")}, [0.5, 0.5]],
)
def test_bad_weights_raise(engine, weights):
    with pytest.raises(ValueError):
        engine.weights_vector(weights)


@pytest.mark.parametrize("confidence, horizon", [(0.5, 1), (0.9999, 1), (0.99, 0), (0.99, 13), (0.99, 2.5)])
def test_out_of_range_parameters_raise(engine, confidence, horizon):
    with pytest.raises(ValueError):
        engine.portfolio({"NVDA": 1.0}, confidence, horizon)


def test_unknown_model_raises(engine):
    with pytest.raises(ValueError):
        engine.portfolio({"NVDA": 1.0}, model_names=["garch"])


# --------------------------------------------------------------------------- Zeitreihen und Kennzahlen


def test_series_start_at_100_and_single_asset_follows_its_price(engine, ds):
    w = engine.weights_vector({"TSM": 1.0})
    value, drawdown = engine.series(w)
    prices = ds.prices["TSM"].to_numpy()
    assert len(value) == len(engine.dates) == len(prices)
    assert value[0] == pytest.approx(100.0) and drawdown[0] == 0.0
    np.testing.assert_allclose(value, 100 * prices / prices[0], rtol=1e-10)
    assert drawdown.max() <= 1e-12 and drawdown.min() < 0


def test_drawdown_kpis_match_stress_module(engine, ds):
    for seed in range(25):
        w = np.eye(len(TICKERS))[seed % len(TICKERS)] if seed % 5 == 0 else random_weights(seed)
        value, drawdown = engine.series(w)
        kpis = engine.kpis(w, value, drawdown)
        ref = stress.max_drawdown(ds.log_returns, w)
        assert kpis["max_drawdown"] == pytest.approx(ref["Drawdown"], abs=1e-12)
        assert kpis["max_drawdown_trough"] == ref["Tief"]
        assert (kpis["max_drawdown_recovered"] or "noch nicht") == ref["Erholt am"]
        assert kpis["max_drawdown_peak"] <= kpis["max_drawdown_trough"]
        assert kpis["volatility"] == pytest.approx(ds.portfolio_volatility(w) * np.sqrt(252))
        assert kpis["total_return"] == pytest.approx(value[-1] / 100 - 1)


def test_portfolio_response_is_json_ready(engine):
    import json

    out = engine.portfolio({"NVDA": 18, "MSFT": 12, "PLTR": 8}, 0.95, 5, ["student_t", "historical"])
    text = json.dumps(out, allow_nan=False)
    assert json.loads(text) == out
    assert sum(out["weights"].values()) == pytest.approx(1.0, abs=1e-5)
    assert len(out["value"]) == len(out["drawdown"]) == out["points"] == len(engine.dates)
    assert set(out["risk"]) == {"student_t", "historical"}
    assert "value" not in engine.portfolio({"NVDA": 1}, with_series=False)


def test_overview_contains_all_presets_and_consistent_meta(engine):
    out = engine.overview()
    meta = out["meta"]
    assert set(out["portfolios"]) == set(PORTFOLIOS)
    assert out["confidence"] == CONFIG.default_confidence and out["horizon"] == CONFIG.default_horizon
    assert out["dates"] == engine.dates and meta["points"] == len(engine.dates) == meta["trading_days"] + 1
    assert [t["symbol"] for t in meta["tickers"]] == TICKERS
    assert all(t["name"] == NAMES[t["symbol"]] and t["group"] == GROUPS[t["symbol"]] for t in meta["tickers"])
    assert meta["limits"]["horizon"] == [1, CONFIG.max_horizon]
    for preset in meta["presets"]:
        assert sum(preset["weights"].values()) == pytest.approx(1.0)
        assert out["portfolios"][preset["id"]]["weights"] == pytest.approx(preset["weights"])


def test_groups_match_stress_sectors():
    assert {t for t, g in GROUPS.items() if g == "chips"} == set(stress.SEMIS)
    assert {t for t, g in GROUPS.items() if g == "platforms"} == set(stress.PLATFORMS)


def test_warm_precomputes_main_model(ds):
    small = RiskEngine(ds, EngineConfig(n_sims=300, chunk=300, max_horizon=3, default_horizon=2))
    small.warm()
    assert list(small._paths) == [MAIN_MODEL] and small._paths[MAIN_MODEL].shape == (3, 300, len(TICKERS))


@pytest.mark.parametrize("kwargs", [{"max_horizon": 3}, {"default_confidence": 0.999}, {"n_sims": 0}, {"cache_models": 0}])
def test_inconsistent_config_raises(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**kwargs)


# --------------------------------------------------------------------------- Konfidenzstufen und Kurven


def test_confidence_steps_cover_the_slider_range(engine):
    steps = engine.confidence_steps()
    assert steps[0] == 0.9 and steps[-1] == 0.995 and len(steps) == 20
    assert CONFIG.default_confidence in steps
    assert engine.horizons() == list(range(1, CONFIG.max_horizon + 1))
    coarse = RiskEngine(engine.ds, EngineConfig(n_sims=100, chunk=100, max_confidence=0.99, confidence_step=0.01))
    assert coarse.confidence_steps() == [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]


@pytest.mark.parametrize("kwargs", [{"confidence_step": 0.004}, {"confidence_step": 0.0}, {"confidence_step": 0.0005}, {"default_confidence": 0.992}])
def test_confidence_grid_must_fit_the_limits(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**kwargs)


@pytest.mark.parametrize("model", ["student_t", "historical", "bootstrap"])
def test_risk_curve_equals_risk_at_every_step(engine, model):
    w = random_weights(7)
    curve = engine.risk_curve(w, 5, model)
    assert curve["confidence"] == engine.confidence_steps()
    for i, confidence in enumerate(curve["confidence"]):
        single = engine.risk(w, confidence, 5, [model])[model]
        assert (curve["var"][i], curve["es"][i], curve["n"]) == (single["var"], single["es"], single["n"])
        assert [curve["var_low"][i], curve["var_high"][i]] == single["var_ci"]
    assert curve["var"] == sorted(curve["var"])  # höhere Konfidenz, höherer VaR
    assert all(low <= var <= high for low, var, high in zip(curve["var_low"], curve["var"], curve["var_high"]))


def test_curve_response_is_json_ready_and_matches_portfolio(engine):
    import json

    out = engine.curve({"NVDA": 3, "MSFT": 1}, 7)
    assert json.loads(json.dumps(out, allow_nan=False)) == out
    assert out["weights"]["NVDA"] == pytest.approx(0.75) and out["horizon"] == 7 and out["model"] == MAIN_MODEL
    assert out["data_end"] == engine.dates[-1] and out["ci_level"] == 0.95 and out["n"] == CONFIG.n_sims
    i = out["confidence"].index(0.975)
    single = engine.portfolio({"NVDA": 3, "MSFT": 1}, 0.975, 7, with_series=False)["risk"][MAIN_MODEL]
    assert (out["var"][i], out["es"][i]) == (single["var"], single["es"])
    assert [out["var_low"][i], out["var_high"][i]] == single["var_ci"]
    assert engine.curve({"NVDA": 1})["horizon"] == CONFIG.default_horizon
    assert engine.curve({"NVDA": 1}, 2, "historical")["model"] == "historical"
    for bad in (lambda: engine.curve({"NVDA": 1}, 99), lambda: engine.curve({"XXX": 1}), lambda: engine.curve({"NVDA": 1}, 2, "garch")):
        with pytest.raises(ValueError):
            bad()


# --------------------------------------------------------------------------- Startdaten der Dashboard-Seite


def test_snapshot_lets_the_page_rebuild_series_and_kpis(engine):
    snapshot = engine.snapshot(generated="2026-01-01T00:00:00+00:00")
    assert snapshot["schema"] == 2 and snapshot["generated"] == "2026-01-01T00:00:00+00:00"
    assert snapshot["meta"] == engine.meta() and snapshot["dates"] == engine.dates
    assert list(snapshot["growth"]) == TICKERS
    growth = np.array([snapshot["growth"][t] for t in TICKERS]).T
    assert growth.shape == (len(engine.dates), len(TICKERS)) and (growth[0] == 1).all()
    cov = np.array(snapshot["cov"])
    for seed in (0, 1):
        w = random_weights(seed)
        value, drawdown = engine.series(w)
        kpis = engine.kpis(w, value, drawdown)
        rebuilt = 100.0 * growth @ w  # so rechnet die Seite
        np.testing.assert_allclose(rebuilt, value, atol=1e-3)
        np.testing.assert_allclose(rebuilt / np.maximum.accumulate(rebuilt) - 1, drawdown, atol=1e-5)
        years = snapshot["meta"]["trading_days_per_year"]
        assert np.sqrt(w @ cov @ w * years) == pytest.approx(kpis["volatility"], abs=1e-7)
        assert (rebuilt[-1] / 100) ** (years / snapshot["meta"]["trading_days"]) - 1 == pytest.approx(kpis["cagr"], abs=1e-5)


def test_snapshot_grid_matches_risk_for_every_preset(engine):
    grid = engine.snapshot()["grid"]
    assert grid["model"] == MAIN_MODEL and grid["ci_level"] == 0.95
    assert grid["confidence"] == engine.confidence_steps() and grid["horizon"] == engine.horizons()
    for name, weights in PORTFOLIOS.items():
        w = engine.weights_vector(dict(zip(TICKERS, weights)))
        cells = {key: np.array(grid["risk"][name][key]) for key in ("var", "var_low", "var_high", "es")}
        assert all(a.shape == (len(grid["horizon"]), len(grid["confidence"])) for a in cells.values())
        assert (np.diff(cells["var"], axis=1) >= 0).all() and (cells["es"] >= cells["var"]).all()
        assert (cells["var_low"] <= cells["var"]).all() and (cells["var"] <= cells["var_high"]).all()
        for h, c in ((0, 0), (4, 11), (len(grid["horizon"]) - 1, len(grid["confidence"]) - 1)):
            ref = engine.portfolio(w, grid["confidence"][c], grid["horizon"][h], with_series=False)["risk"][MAIN_MODEL]
            assert (cells["var"][h, c], cells["es"][h, c]) == (ref["var"], ref["es"])
            assert [cells["var_low"][h, c], cells["var_high"][h, c]] == ref["var_ci"]


def test_snapshot_file_is_valid_javascript_data(engine, tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "build_snapshot.py"
    spec = importlib.util.spec_from_file_location("build_snapshot", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    snapshot = engine.snapshot()
    out = tmp_path / "daten.js"
    module.write_snapshot(snapshot, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("/*") and text.rstrip().endswith(";")
    start = text.index("window.RISK_SNAPSHOT=") + len("window.RISK_SNAPSHOT=")
    assert json.loads(text[start:].rstrip().rstrip(";")) == snapshot
    assert "NaN" not in text and "Infinity" not in text
