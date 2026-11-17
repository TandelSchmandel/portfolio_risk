"""HTTP-Schnittstelle für das Dashboard (FastAPI).

Start (lokal):

    pip install -r requirements-api.txt
    uvicorn portfolio_risk.api:create_app --factory --app-dir src --port 8000

Die Schnittstelle ist öffentlich und nur lesend: Sie rechnet mit den Kursen aus ``data/raw`` und
nimmt keine Daten von Besuchern entgegen außer Gewichten, Konfidenz und Horizont in der Adresse.

Endpunkte (Beschreibung und Ausprobieren unter ``/docs``):

    GET /api/health     Lebenszeichen, Datenstand, Szenarien vorgerechnet?
    GET /api/meta       Ticker, vorgegebene Portfolios, Modelle, erlaubte Wertebereiche
    GET /api/snapshot   Startdaten der Dashboard-Seite (Kursverläufe, VaR/ES der vorgegebenen Portfolios)
    GET /api/curve      VaR/ES für eigene Gewichte über alle Konfidenzstufen bei einem Horizont
    GET /api/portfolio  Kennzahlen, Zeitreihen und VaR/ES für eigene Gewichte, alle Modelle wählbar
    GET /api/overview   die drei vorgegebenen Portfolios in der Form von /api/portfolio

Einstellungen über Umgebungsvariablen:

    RISK_DATA_DIR         Ordner mit den Kurs-CSVs (Standard: data/raw im Repository)
    RISK_ALLOWED_ORIGINS  erlaubte Herkunftsseiten für Browser-Anfragen, durch Komma getrennt (Standard: *)
    RISK_N_SIMS           Simulationen je Modell und Horizont (Standard: 100000)
    RISK_RATE_LIMIT       Anfragen je Minute und Besucher für die Rechen-Endpunkte (Standard: 300, 0 = aus)
"""

from __future__ import annotations

import gzip
import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response

from portfolio_risk.dataset import build_dataset
from portfolio_risk.engine import RISK_MODELS, EngineConfig, RiskEngine
from portfolio_risk.portfolios import TICKERS

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CACHE_SECONDS = 300
LIMITED_PATHS = ("/api/portfolio", "/api/overview", "/api/curve")


def engine_from_env() -> RiskEngine:
    data_dir = Path(os.environ.get("RISK_DATA_DIR", DEFAULT_DATA_DIR))
    config = EngineConfig(n_sims=int(os.environ.get("RISK_N_SIMS", EngineConfig.n_sims)))
    return RiskEngine(build_dataset(data_dir, TICKERS), config)


def parse_weights(text: str) -> dict[str, float]:
    """Liest ``NVDA:18,AVGO:10`` in ein Dict. Die Skala ist frei, der Rechenkern normiert auf Summe 1."""
    weights: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ticker, sep, value = part.partition(":")
        ticker = ticker.strip().upper()
        try:
            number = float(value)
        except ValueError:
            number = float("nan")
        if not sep or not ticker or number != number:
            raise ValueError(f"Gewicht {part!r} hat nicht die Form TICKER:ZAHL")
        if ticker in weights:
            raise ValueError(f"Ticker {ticker} kommt mehrfach vor")
        weights[ticker] = number
    if not weights:
        raise ValueError("Keine Gewichte angegeben")
    return weights


def parse_models(text: str | None) -> list[str] | None:
    if text is None or not text.strip():
        return None
    if text.strip().lower() == "all":
        return list(RISK_MODELS)
    return [m.strip().lower() for m in text.split(",") if m.strip()]


class RateLimiter:
    """Gleitendes Fenster je Besucher. Schützt die Rechen-Endpunkte vor Dauerfeuer, mehr nicht.

    Die Zählung liegt nur im Arbeitsspeicher. Besucher, deren letzte Anfrage länger als ein Fenster
    zurückliegt, werden bei der nächsten Anfrage nach Ablauf eines Fensters verworfen (Datensparsamkeit:
    IP-Adressen bleiben nicht liegen).
    """

    def __init__(self, per_minute: int, window: float = 60.0, max_clients: int = 10_000):
        self.per_minute = per_minute
        self.window = window
        self.max_clients = max_clients
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._pruned = time.monotonic()

    def clients(self) -> int:
        with self._lock:
            return len(self._hits)

    def allow(self, client: str, now: float | None = None) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            if now - self._pruned >= self.window or len(self._hits) > self.max_clients:
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < self.window}
                self._pruned = now
            hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True


def create_app(
    engine: RiskEngine | None = None,
    *,
    allowed_origins: list[str] | None = None,
    rate_limit: int | None = None,
    warm: bool = True,
) -> FastAPI:
    engine = engine or engine_from_env()
    if allowed_origins is None:
        allowed_origins = [o.strip() for o in os.environ.get("RISK_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    limiter = RateLimiter(int(os.environ.get("RISK_RATE_LIMIT", 300)) if rate_limit is None else rate_limit)
    state = {"warm": not warm}
    snapshot_lock = threading.Lock()

    def snapshot_body() -> tuple[bytes, bytes]:
        """Startdaten als JSON, roh und gepackt. Einmal je Prozess berechnet: Die Kurse ändern sich nur mit einem Neustart."""
        with snapshot_lock:
            if "snapshot" not in state:
                raw = json.dumps(engine.snapshot(), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
                state["snapshot"] = (raw, gzip.compress(raw, compresslevel=6, mtime=0))
            return state["snapshot"]

    def warm_up() -> None:
        engine.warm()
        snapshot_body()
        state["warm"] = True

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if warm:
            # Im Hintergrund, damit der Dienst sofort antwortet. Fehlende Szenarien rechnet eine Anfrage selbst.
            threading.Thread(target=warm_up, name="szenarien", daemon=True).start()
        yield

    app = FastAPI(
        title="Portfolio-Risiko",
        description=(
            "Rechnet Wertentwicklung, Drawdown, Value at Risk und Expected Shortfall für ein Portfolio aus zehn "
            "KI-Aktien mit frei wählbaren Gewichten, Konfidenz und Horizont. Grundlage sind tägliche Kurse "
            "seit November 2021. VaR und ES sind positive Verluste als Anteil des Portfoliowerts. Keine Anlageberatung."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.add_middleware(GZipMiddleware, minimum_size=500)
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET"], max_age=86400)

    @app.middleware("http")
    async def limit_and_cache(request: Request, call_next):
        path = request.url.path
        if request.method == "GET" and path.startswith(LIMITED_PATHS):
            forwarded = request.headers.get("x-forwarded-for", "")
            client = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unbekannt")
            if not limiter.allow(client):
                return JSONResponse(
                    {"detail": "Zu viele Anfragen. Bitte kurz warten."}, status_code=429, headers={"Retry-After": "30"}
                )
        response = await call_next(request)
        if request.method == "GET" and path.startswith("/api/") and response.status_code == 200:
            response.headers["Cache-Control"] = "no-store" if path == "/api/health" else f"public, max-age={CACHE_SECONDS}"
        return response

    def checked(call):
        try:
            return call()
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    weights_query = Query(
        ...,
        description="Gewichte als `TICKER:ZAHL`, durch Komma getrennt, z. B. `NVDA:50,MSFT:30,AMZN:20`. "
        "Die Skala ist frei, es wird auf 100 % normiert. Fehlende Ticker zählen als 0.",
        max_length=400,
    )
    confidence_query = Query(None, description="Konfidenzniveau als Anteil, z. B. 0.99. Standard und Grenzen stehen in /api/meta.")
    horizon_query = Query(None, description="Horizont in Handelstagen, z. B. 10.")
    models_query = Query(None, description=f"Modelle, durch Komma getrennt, oder `all`. Erlaubt: {', '.join(RISK_MODELS)}. Standard: student_t.")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/api/health", summary="Lebenszeichen")
    def health():
        return {"status": "ok", "data_end": engine.dates[-1], "warm": state["warm"]}

    @app.get("/api/meta", summary="Ticker, vorgegebene Portfolios, Modelle und erlaubte Wertebereiche")
    def meta():
        return engine.meta()

    @app.get(
        "/api/snapshot",
        summary="Startdaten der Dashboard-Seite",
        description="Metadaten, Datumsachse, Kursverlauf je Titel (Kurs geteilt durch den Kurs am Starttag), Kovarianzmatrix und "
        "VaR/ES des Hauptmodells für die vorgegebenen Portfolios über alle Horizonte und Konfidenzstufen. Wertentwicklung und "
        "Drawdown rechnet die Seite daraus selbst. Derselbe Aufbau wie die Datei aus `scripts/build_snapshot.py`.",
    )
    def snapshot(request: Request):
        raw, packed = snapshot_body()
        if "gzip" in request.headers.get("accept-encoding", "").lower():
            return Response(packed, media_type="application/json", headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})
        return Response(raw, media_type="application/json")

    @app.get("/api/curve", summary="VaR und ES für eigene Gewichte über alle Konfidenzstufen bei einem Horizont")
    def curve(
        weights: str = weights_query,
        horizon: int | None = horizon_query,
        model: str | None = Query(None, description=f"Ein Modell. Erlaubt: {', '.join(RISK_MODELS)}. Standard: student_t."),
    ):
        return checked(lambda: engine.curve(parse_weights(weights), horizon, model.strip().lower() if model else None))

    @app.get("/api/overview", summary="Startzustand: Metadaten, Datumsachse und die drei vorgegebenen Portfolios")
    def overview(confidence: float | None = confidence_query, horizon: int | None = horizon_query, models: str | None = models_query):
        return checked(lambda: engine.overview(confidence, horizon, parse_models(models)))

    @app.get("/api/portfolio", summary="Kennzahlen, Zeitreihen und VaR/ES für eigene Gewichte")
    def portfolio(
        weights: str = weights_query,
        confidence: float | None = confidence_query,
        horizon: int | None = horizon_query,
        models: str | None = models_query,
        series: bool = Query(True, description="`false` lässt die Zeitreihen weg (für Änderungen an Konfidenz oder Horizont)."),
    ):
        return checked(lambda: engine.portfolio(parse_weights(weights), confidence, horizon, parse_models(models), with_series=series))

    return app
