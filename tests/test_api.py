"""HTTP-Schnittstelle: Antworten, Fehlerfälle, Schutz vor Dauerfeuer."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")

from fastapi.testclient import TestClient  # noqa: E402

from portfolio_risk.api import RateLimiter, create_app, parse_models, parse_weights  # noqa: E402
from portfolio_risk.dataset import build_dataset  # noqa: E402
from portfolio_risk.engine import RISK_MODELS, EngineConfig, RiskEngine  # noqa: E402
from portfolio_risk.portfolios import PORTFOLIOS, TICKERS  # noqa: E402

from test_engine import write_prices  # noqa: E402


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    directory = tmp_path_factory.mktemp("kurse_api")
    write_prices(directory)
    return RiskEngine(build_dataset(directory, TICKERS), EngineConfig(n_sims=4_000, chunk=4_000))


@pytest.fixture(scope="module")
def client(engine):
    with TestClient(create_app(engine, rate_limit=0, warm=False)) as c:
        yield c


def test_health_and_meta(client, engine):
    health = client.get("/api/health")
    assert health.status_code == 200 and health.json() == {"status": "ok", "data_end": engine.dates[-1], "warm": True}
    assert health.headers["cache-control"] == "no-store"
    meta = client.get("/api/meta")
    assert meta.json() == engine.meta()
    assert meta.headers["cache-control"].startswith("public")


def test_overview_matches_engine(client, engine):
    response = client.get("/api/overview", params={"confidence": 0.95, "horizon": 5})
    body = response.json()
    assert response.status_code == 200
    assert body["confidence"] == 0.95 and body["horizon"] == 5
    assert set(body["portfolios"]) == set(PORTFOLIOS)
    assert body == engine.overview(0.95, 5)


def test_portfolio_with_named_weights(client, engine):
    response = client.get("/api/portfolio", params={"weights": "nvda:50, MSFT:30,AMZN:20", "confidence": 0.975, "horizon": 3})
    body = response.json()
    assert response.status_code == 200
    assert body["weights"]["NVDA"] == pytest.approx(0.5) and body["weights"]["TSM"] == 0
    assert len(body["value"]) == len(body["drawdown"]) == body["points"]
    assert body == engine.portfolio({"NVDA": 50, "MSFT": 30, "AMZN": 20}, 0.975, 3)


def test_portfolio_without_series_and_with_all_models(client):
    body = client.get("/api/portfolio", params={"weights": "PLTR:1", "series": "false", "models": "all"}).json()
    assert "value" not in body and "drawdown" not in body
    assert list(body["risk"]) == list(RISK_MODELS)
    assert body["confidence"] == 0.99 and body["horizon"] == 10


def test_curve_matches_engine(client, engine):
    response = client.get("/api/curve", params={"weights": "NVDA:3,msft:1", "horizon": 4})
    body = response.json()
    assert response.status_code == 200
    assert body == engine.curve({"NVDA": 3, "MSFT": 1}, 4)
    assert body["confidence"] == engine.confidence_steps() and len(body["var"]) == len(body["es"]) == 20
    assert client.get("/api/curve", params={"weights": "NVDA:1"}).json()["horizon"] == engine.config.default_horizon
    assert client.get("/api/curve", params={"weights": "NVDA:1", "model": "Historical"}).json()["model"] == "historical"


@pytest.mark.parametrize("params", [{}, {"weights": "XXX:1"}, {"weights": "NVDA:1", "horizon": 0}, {"weights": "NVDA:1", "model": "garch"}, {"weights": "NVDA:1", "model": "fhs,normal"}])
def test_invalid_curve_requests_return_422(client, params):
    response = client.get("/api/curve", params=params)
    assert response.status_code == 422 and response.json()["detail"]


def test_snapshot_is_served_packed_and_matches_engine(client, engine):
    response = client.get("/api/snapshot")
    body = response.json()
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip" and "accept-encoding" in response.headers["vary"].lower()
    assert response.headers["cache-control"].startswith("public")
    assert body == engine.snapshot(generated=body["generated"])
    plain = client.get("/api/snapshot", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers and plain.json() == body
    assert client.get("/api/snapshot").json()["generated"] == body["generated"]  # einmal berechnet, danach aus dem Speicher


@pytest.mark.parametrize(
    "params",
    [
        {"weights": "XXX:1"},
        {"weights": "NVDA"},
        {"weights": "NVDA:abc"},
        {"weights": "NVDA:1,NVDA:2"},
        {"weights": "NVDA:-1,MSFT:2"},
        {"weights": "NVDA:0"},
        {"weights": " , "},
        {"weights": "NVDA:1", "confidence": 0.5},
        {"weights": "NVDA:1", "horizon": 21},
        {"weights": "NVDA:1", "horizon": 2.5},
        {"weights": "NVDA:1", "models": "garch"},
        {},
    ],
)
def test_invalid_requests_return_422(client, params):
    response = client.get("/api/portfolio", params=params)
    assert response.status_code == 422
    assert response.json()["detail"]


def test_cors_allows_configured_origin_only(engine):
    with TestClient(create_app(engine, allowed_origins=["https://example.org"], rate_limit=0, warm=False)) as c:
        ok = c.get("/api/meta", headers={"Origin": "https://example.org"})
        other = c.get("/api/meta", headers={"Origin": "https://fremd.example"})
    assert ok.headers["access-control-allow-origin"] == "https://example.org"
    assert "access-control-allow-origin" not in other.headers


def test_rate_limit_returns_429_per_client(engine):
    with TestClient(create_app(engine, rate_limit=3, warm=False)) as c:
        codes = [c.get("/api/portfolio", params={"weights": "NVDA:1", "series": "false"}).status_code for _ in range(5)]
        other = c.get("/api/portfolio", params={"weights": "NVDA:1", "series": "false"}, headers={"X-Forwarded-For": "203.0.113.9"})
        health = c.get("/api/health")
    assert codes == [200, 200, 200, 429, 429]
    assert other.status_code == 200 and health.status_code == 200


def test_rate_limit_covers_curve_but_not_snapshot(engine):
    with TestClient(create_app(engine, rate_limit=2, warm=False)) as c:
        codes = [c.get("/api/curve", params={"weights": "NVDA:1", "horizon": 1}).status_code for _ in range(3)]
        snapshots = [c.get("/api/snapshot").status_code for _ in range(4)]
    assert codes == [200, 200, 429] and snapshots == [200] * 4


def test_rate_limiter_window_slides():
    limiter = RateLimiter(per_minute=2, window=60)
    assert limiter.allow("a", now=0) and limiter.allow("a", now=1)
    assert not limiter.allow("a", now=30)
    assert limiter.allow("b", now=30)
    assert limiter.allow("a", now=61)
    assert RateLimiter(per_minute=0).allow("a")


def test_rate_limiter_forgets_idle_clients():
    limiter = RateLimiter(per_minute=5, window=60)
    limiter._pruned = 0.0
    assert limiter.allow("a", now=0) and limiter.allow("b", now=10)
    assert limiter.allow("c", now=65)  # ein Fenster ist um: "a" fällt weg, "b" bleibt
    assert limiter.clients() == 2
    assert limiter.allow("d", now=200)
    assert limiter.clients() == 1


def test_parse_helpers():
    assert parse_weights("NVDA:18, msft:12.5") == {"NVDA": 18.0, "MSFT": 12.5}
    assert parse_models(None) is None and parse_models(" ") is None
    assert parse_models("ALL") == list(RISK_MODELS)
    assert parse_models("fhs, Normal") == ["fhs", "normal"]


def test_warm_up_runs_in_background(engine, monkeypatch):
    import time

    calls = []
    original = engine.snapshot
    monkeypatch.setattr(engine, "snapshot", lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))
    with TestClient(create_app(engine, rate_limit=0)) as c:
        for _ in range(400):
            if c.get("/api/health").json()["warm"]:
                break
            time.sleep(0.05)
        assert c.get("/api/health").json()["warm"]
        assert c.get("/api/snapshot").status_code == 200
    assert calls == [1]  # beim Vorrechnen erzeugt, die Anfrage kommt aus dem Speicher
