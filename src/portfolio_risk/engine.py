"""Rechenkern für das Dashboard: Kennzahlen für beliebige Gewichte in Millisekunden.

Die Simulationen aus ``models.py`` hängen nicht von den Gewichten ab. Der Rechenkern simuliert
deshalb je Modell einmal die Renditen **je Titel** für alle Horizonte (``models.scenario_paths``)
und hält sie im Speicher. Ein Portfolio ist danach nur noch ein Matrix-Vektor-Produkt, VaR und ES
kommen aus ``risk.var_es_with_ci``. Alle Definitionen sind dieselben wie in Schritt 2:

* VaR und ES sind positive Verluste als Anteil des Portfoliowerts, Simulation ohne Drift.
* Wertentwicklung und Drawdown sind Buy-and-hold ab dem ersten Tag (wie ``stress.max_drawdown``).
* Die Volatilität ist ``sqrt(w' Cov w)`` aus der Kovarianzmatrix des Datenmoduls.

Mit gleicher Simulationszahl und gleichem Seed liefert der Rechenkern exakt die Werte der Funktionen
``models.simulate_*``. Der einzige Unterschied zum Bericht ist die Zahl der Simulationen (hier 100 000
statt 1 000 000), damit der Dienst mit wenig Speicher auskommt. Das Konfidenzintervall des VaR wird mitgeliefert.

Drei Sichten auf dasselbe Portfolio:

* ``portfolio``  Kennzahlen, Zeitreihen und VaR/ES für eine Konfidenz und einen Horizont (alle Modelle wählbar)
* ``curve``      VaR/ES für alle Konfidenzstufen des Reglers bei einem Horizont (ein Modell)
* ``snapshot``   Startdaten der Dashboard-Seite: Kursverläufe je Titel und VaR/ES der vorgegebenen Portfolios
                 für alle Horizonte und Konfidenzstufen. Wertentwicklung und Drawdown rechnet die Seite daraus selbst.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

import numpy as np

from portfolio_risk import models
from portfolio_risk.dataset import TRADING_DAYS, RiskDataset
from portfolio_risk.portfolios import GROUPS, NAMES, PORTFOLIOS
from portfolio_risk.risk import var_es_curve

MAIN_MODEL = "student_t"
RISK_MODELS = ("student_t", "historical", "normal", "bootstrap", "fhs")
SCHEMA_VERSION = 2  # Aufbau der Antworten; die Dashboard-Seite prüft diese Zahl
CI_LEVEL = 0.95  # Niveau des Konfidenzintervalls für den VaR


@dataclass(frozen=True)
class EngineConfig:
    n_sims: int = 100_000
    seed: int = 20261116  # derselbe Seed wie scripts/risk_report_v2.py
    min_horizon: int = 1
    max_horizon: int = 20
    min_confidence: float = 0.90
    max_confidence: float = 0.995
    confidence_step: float = 0.005  # Raster des Konfidenz-Reglers, ein Vielfaches von 0,001
    default_confidence: float = 0.99
    default_horizon: int = 10
    default_preset: str = "A_Hauptportfolio"
    # Speicher: ein Pfadmodell belegt max_horizon x n_sims x Titel x 4 Byte (20 x 100 000 x 10: 80 MB),
    # eine Bootstrap-Matrix n_sims x Titel x 4 Byte (4 MB). Die Voreinstellung passt in 512 MB.
    cache_models: int = 2
    cache_matrices: int = 8
    chunk: int = 20_000  # Blockgröße beim Bootstrap, hält den Speicherbedarf beim Rechnen klein

    def __post_init__(self):
        if not (1 <= self.min_horizon <= self.default_horizon <= self.max_horizon):
            raise ValueError("Horizont: Untergrenze, Standardwert und Obergrenze passen nicht zusammen")
        if not (0.5 < self.min_confidence <= self.default_confidence <= self.max_confidence < 1.0):
            raise ValueError("Konfidenz: Untergrenze, Standardwert und Obergrenze passen nicht zusammen")
        low, high, default, step = (self.min_confidence * 1000, self.max_confidence * 1000, self.default_confidence * 1000, self.confidence_step * 1000)
        on_grid = all(abs(x - round(x)) < 1e-6 for x in (low, high, default, step))
        if not on_grid or round(step) < 1 or round(high - low) % round(step) or round(default - low) % round(step):
            raise ValueError("Konfidenz: Grenzen und Standardwert müssen auf dem Raster von confidence_step liegen")
        if min(self.n_sims, self.chunk, self.cache_models, self.cache_matrices) < 1:
            raise ValueError("n_sims, chunk, cache_models und cache_matrices müssen positiv sein")


class RiskEngine:
    """Hält Daten und Szenarien im Speicher und rechnet Portfolios mit beliebigen Gewichten."""

    def __init__(self, dataset: RiskDataset, config: EngineConfig | None = None):
        self.ds = dataset
        self.config = config or EngineConfig()
        self.tickers = dataset.tickers
        lr = dataset.log_returns
        # Datumsachse der Zeitreihen: alle Kurstage. Der erste ist der Starttag (Wert 100), danach ein Tag je Rendite.
        self.dates = [d.date().isoformat() for d in dataset.prices.index]
        if len(self.dates) != len(lr) + 1:
            raise ValueError("Kurse und Renditen passen nicht zusammen")
        self.nu = round(models.fit_t_degrees_of_freedom(lr, cov=dataset.cov), 2)
        # Wachstumsfaktor je Titel seit dem Starttag: 1 am Starttag, danach exp(Summe der Log-Renditen)
        self._growth = np.vstack([np.ones(len(self.tickers)), np.exp(lr.cumsum().to_numpy())])
        self._cov = dataset.cov.to_numpy()
        self._paths: OrderedDict[str, np.ndarray] = OrderedDict()  # Pfadmodell -> (Horizont, n_sims, Titel)
        self._matrices: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()  # (Modell, Horizont) -> (n_sims, Titel)
        self._historical: dict[int, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._simulation_lock = threading.Lock()  # nie zwei Simulationen gleichzeitig (Speicher, Rechenzeit)

    # ------------------------------------------------------------------ Eingaben prüfen

    def weights_vector(self, weights: Mapping[str, float] | Iterable[float]) -> np.ndarray:
        """Gewichte als Vektor in Ticker-Reihenfolge, auf Summe 1 normiert.

        Erlaubt sind nicht negative Zahlen in beliebiger Skala (Anteile oder Prozent). Fehlende
        Ticker zählen als 0. Leerverkäufe und Hebel gibt es nicht.
        """
        if isinstance(weights, Mapping):
            unknown = [t for t in weights if t not in self.tickers]
            if unknown:
                raise ValueError(f"Unbekannte Ticker {unknown}, erlaubt: {self.tickers}")
            w = np.array([float(weights.get(t, 0.0)) for t in self.tickers])
        else:
            w = np.asarray(list(weights), dtype=float)
            if w.shape != (len(self.tickers),):
                raise ValueError(f"{w.size} Gewichte für {len(self.tickers)} Ticker")
        if not np.all(np.isfinite(w)):
            raise ValueError("Gewichte müssen endliche Zahlen sein")
        if (w < 0).any():
            raise ValueError("Gewichte dürfen nicht negativ sein")
        total = w.sum()
        if total <= 0:
            raise ValueError("Mindestens ein Gewicht muss größer als 0 sein")
        return w / total

    def check_confidence(self, confidence: float) -> float:
        c = float(confidence)
        if not (self.config.min_confidence <= c <= self.config.max_confidence):
            raise ValueError(f"Konfidenz muss zwischen {self.config.min_confidence} und {self.config.max_confidence} liegen")
        return c

    def check_horizon(self, horizon: int) -> int:
        h = int(horizon)
        if h != horizon or not (self.config.min_horizon <= h <= self.config.max_horizon):
            raise ValueError(f"Horizont muss eine ganze Zahl von {self.config.min_horizon} bis {self.config.max_horizon} Handelstagen sein")
        return h

    def check_models(self, names: Iterable[str] | None) -> tuple[str, ...]:
        if names is None:
            return (MAIN_MODEL,)
        names = tuple(dict.fromkeys(names))
        unknown = [n for n in names if n not in RISK_MODELS]
        if unknown or not names:
            raise ValueError(f"Unbekannte Modelle {unknown}, erlaubt: {list(RISK_MODELS)}")
        return names

    def confidence_steps(self) -> list[float]:
        """Konfidenzstufen des Reglers, von der Unter- bis zur Obergrenze."""
        cfg = self.config
        low, high, step = (round(x * 1000) for x in (cfg.min_confidence, cfg.max_confidence, cfg.confidence_step))
        return [k / 1000 for k in range(low, high + 1, step)]

    def horizons(self) -> list[int]:
        return list(range(self.config.min_horizon, self.config.max_horizon + 1))

    # ------------------------------------------------------------------ Szenarien

    def scenarios(self, model: str, horizon: int) -> np.ndarray:
        """Simulierte Horizont-Renditen je Titel (n_sims x Titel, float32), einmal berechnet und dann wiederverwendet."""
        if model in models.PATH_MODELS:
            return self._cached(self._paths, model, self.config.cache_models, lambda: self._simulate_paths(model))[horizon - 1]
        return self._cached(self._matrices, (model, horizon), self.config.cache_matrices, lambda: self._simulate_matrix(model, horizon))

    def _cached(self, cache: OrderedDict, key, capacity: int, compute) -> np.ndarray:
        with self._cache_lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
        with self._simulation_lock:
            with self._cache_lock:
                if key in cache:
                    return cache[key]
            value = compute()
            value.setflags(write=False)
            with self._cache_lock:
                cache[key] = value
                while len(cache) > capacity:
                    cache.popitem(last=False)
        return value

    def _simulate_paths(self, model: str) -> np.ndarray:
        cfg = self.config
        kwargs = {"cov": self.ds.cov} if model in ("normal", "student_t") else {}
        if model == "student_t":
            kwargs["nu"] = self.nu
        return models.scenario_paths(model, self.ds.log_returns, cfg.max_horizon, cfg.n_sims, cfg.seed, dtype=np.float32, **kwargs)

    def _simulate_matrix(self, model: str, horizon: int) -> np.ndarray:
        cfg = self.config
        return models.scenario_returns(model, self.ds.log_returns, horizon, cfg.n_sims, cfg.seed, cfg.chunk, dtype=np.float32)

    def historical(self, horizon: int) -> np.ndarray:
        """Echte Buy-and-hold-Renditen je Titel über alle überlappenden Fenster von `horizon` Tagen."""
        cached = self._historical.get(horizon)
        if cached is None:
            cum = self.ds.log_returns.rolling(horizon).sum().dropna().to_numpy()
            cached = np.expm1(cum)
            cached.setflags(write=False)
            self._historical[horizon] = cached
        return cached

    def warm(self, model_names: Iterable[str] = (MAIN_MODEL,)) -> None:
        """Rechnet Szenarien vor, damit die erste Anfrage nicht wartet. Standard: das Hauptmodell, alle Horizonte."""
        for name in model_names:
            if name in models.PATH_MODELS:
                self.scenarios(name, self.config.default_horizon)

    # ------------------------------------------------------------------ Kennzahlen

    def risk(self, w: np.ndarray, confidence: float, horizon: int, model_names: Iterable[str] | None = None) -> dict:
        """VaR und ES je Modell für die Gewichte `w` (bereits normiert)."""
        out = {}
        for name in self.check_models(model_names):
            curve = self.risk_curve(w, horizon, name, [confidence])
            out[name] = {
                "var": curve["var"][0],
                "var_ci": [curve["var_low"][0], curve["var_high"][0]],
                "es": curve["es"][0],
                "n": curve["n"],
            }
        return out

    def risk_curve(self, w: np.ndarray, horizon: int, model: str = MAIN_MODEL, confidences: Iterable[float] | None = None) -> dict:
        """VaR, VaR-Intervall und ES eines Modells für mehrere Konfidenzniveaus (Standard: alle Stufen des Reglers).

        Die Portfolio-Renditen werden einmal gebildet und einmal teilsortiert. Das kostet kaum mehr als
        ein einzelnes Niveau, und die Werte sind dieselben wie bei ``risk``.
        """
        confidences = self.confidence_steps() if confidences is None else list(confidences)
        if model == "historical":
            samples = self.historical(horizon) @ w
            res = var_es_curve(samples, confidences, CI_LEVEL, effective_n=len(samples) / horizon)
        else:
            samples = self.scenarios(model, horizon) @ w.astype(np.float32)
            res = var_es_curve(samples, confidences, CI_LEVEL)
        return {
            "confidence": confidences,
            "var": res["VaR"],
            "var_low": res["VaR KI unten"],
            "var_high": res["VaR KI oben"],
            "es": res["ES"],
            "n": int(res["n"]),
        }

    def series(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Wert (100 am Starttag) und Drawdown je Kurstag, beide Buy-and-hold ohne Umschichtung."""
        value = 100.0 * (self._growth @ w)
        return value, value / np.maximum.accumulate(value) - 1.0

    def kpis(self, w: np.ndarray, value: np.ndarray, drawdown: np.ndarray) -> dict:
        """Kennzahlen der Zeitreihe. Der maximale Drawdown ist derselbe wie in ``stress.max_drawdown``."""
        trough = int(np.argmin(drawdown))
        peak = int(np.argmax(value[: trough + 1]))
        recovered = np.nonzero(value[trough:] >= value[peak])[0]
        growth = float(value[-1] / 100.0)
        n_returns = len(value) - 1
        return {
            "end_value": float(value[-1]),
            "total_return": growth - 1.0,
            "cagr": growth ** (TRADING_DAYS / n_returns) - 1.0,
            "volatility": float(math.sqrt(w @ self._cov @ w) * math.sqrt(TRADING_DAYS)),
            "max_drawdown": float(drawdown[trough]),
            "max_drawdown_peak": self.dates[peak],
            "max_drawdown_trough": self.dates[trough],
            "max_drawdown_recovered": self.dates[trough + int(recovered[0])] if recovered.size else None,
            "current_drawdown": float(drawdown[-1]),
        }

    def portfolio(
        self,
        weights: Mapping[str, float] | Iterable[float],
        confidence: float | None = None,
        horizon: int | None = None,
        model_names: Iterable[str] | None = None,
        with_series: bool = True,
    ) -> dict:
        """Alles zu einem Portfolio: normierte Gewichte, Kennzahlen, Zeitreihen und VaR/ES."""
        cfg = self.config
        w = self.weights_vector(weights)
        c = self.check_confidence(cfg.default_confidence if confidence is None else confidence)
        h = self.check_horizon(cfg.default_horizon if horizon is None else horizon)
        value, drawdown = self.series(w)
        out = {
            "weights": {t: _round(x, 6) for t, x in zip(self.tickers, w)},
            "confidence": c,
            "horizon": h,
            "data_end": self.dates[-1],
            "points": len(self.dates),
            "kpis": _rounded(self.kpis(w, value, drawdown)),
            "risk": _rounded(self.risk(w, c, h, model_names)),
        }
        if with_series:
            out["value"] = np.round(value, 2).tolist()
            out["drawdown"] = np.round(drawdown, 4).tolist()
        return out

    def curve(self, weights: Mapping[str, float] | Iterable[float], horizon: int | None = None, model: str | None = None) -> dict:
        """VaR und ES für eigene Gewichte über alle Konfidenzstufen des Reglers bei einem Horizont."""
        w = self.weights_vector(weights)
        h = self.check_horizon(self.config.default_horizon if horizon is None else horizon)
        (name,) = self.check_models(None if model is None else [model])
        return {
            "weights": {t: _round(x, 6) for t, x in zip(self.tickers, w)},
            "horizon": h,
            "data_end": self.dates[-1],
            "model": name,
            "ci_level": CI_LEVEL,
            **_rounded(self.risk_curve(w, h, name)),
        }

    # ------------------------------------------------------------------ Übersicht

    def meta(self) -> dict:
        cfg = self.config
        return {
            "schema": SCHEMA_VERSION,
            "data_start": self.dates[0],
            "data_end": self.dates[-1],
            "points": len(self.dates),
            "trading_days": len(self.dates) - 1,
            "trading_days_per_year": TRADING_DAYS,
            "tickers": [{"symbol": t, "name": NAMES.get(t, t), "group": GROUPS.get(t)} for t in self.tickers],
            "presets": [
                {"id": name, "key": name.split("_", 1)[0], "weights": dict(zip(self.tickers, weights))}
                for name, weights in PORTFOLIOS.items()
            ],
            "models": list(RISK_MODELS),
            "main_model": MAIN_MODEL,
            "student_t_nu": self.nu,
            "n_sims": cfg.n_sims,
            "ci_level": CI_LEVEL,
            "confidence_steps": self.confidence_steps(),
            "limits": {
                "confidence": [cfg.min_confidence, cfg.max_confidence],
                "horizon": [cfg.min_horizon, cfg.max_horizon],
            },
            "defaults": {"confidence": cfg.default_confidence, "horizon": cfg.default_horizon, "preset": cfg.default_preset},
        }

    def overview(self, confidence: float | None = None, horizon: int | None = None, model_names: Iterable[str] | None = None) -> dict:
        """Startzustand des Dashboards: Metadaten, Datumsachse und die drei vorgegebenen Portfolios."""
        portfolios = {
            name: self.portfolio(dict(zip(self.tickers, weights)), confidence, horizon, model_names)
            for name, weights in PORTFOLIOS.items()
        }
        first = next(iter(portfolios.values()))
        return {
            "meta": self.meta(),
            "dates": self.dates,
            "confidence": first["confidence"],
            "horizon": first["horizon"],
            "portfolios": portfolios,
        }

    def snapshot(self, generated: str | None = None) -> dict:
        """Startdaten der Dashboard-Seite.

        ``growth`` ist der Kurs je Titel geteilt durch den Kurs am Starttag. Daraus rechnet die Seite
        Wertentwicklung (``100 * growth @ w``), Drawdown und Kennzahlen für beliebige Gewichte selbst,
        ``cov`` (tägliche Kovarianz der Log-Renditen) liefert die Volatilität. ``grid`` enthält VaR und ES
        des Hauptmodells für die vorgegebenen Portfolios: ``grid.risk[<Portfolio>].var[<Index Horizont>][<Index Konfidenz>]``.
        Für eigene Gewichte fragt die Seite ``curve`` ab.
        """
        horizons = self.horizons()
        risk = {}
        for name, weights in PORTFOLIOS.items():
            w = self.weights_vector(dict(zip(self.tickers, weights)))
            curves = [self.risk_curve(w, h) for h in horizons]
            risk[name] = {key: [_rounded(c[key]) for c in curves] for key in ("var", "var_low", "var_high", "es")}
        return {
            "schema": SCHEMA_VERSION,
            "generated": generated or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "meta": self.meta(),
            "dates": self.dates,
            "growth": {t: np.round(self._growth[:, i], 5).tolist() for i, t in enumerate(self.tickers)},
            "cov": np.round(self._cov, 12).tolist(),
            "grid": {
                "model": MAIN_MODEL,
                "ci_level": CI_LEVEL,
                "confidence": self.confidence_steps(),
                "horizon": horizons,
                "risk": risk,
            },
        }


def _round(x: float, digits: int = 6) -> float:
    return round(float(x), digits)


def _rounded(obj):
    """Rundet alle Fließkommazahlen in verschachtelten Dicts und Listen (kompaktes JSON)."""
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_rounded(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return _round(obj)
    return obj
