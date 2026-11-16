"""Simulationsmodelle für Portfolio-Renditen über einen Horizont von h Handelstagen.

Gemeinsame Konventionen aller Modelle:

* Eingabe sind tägliche **Log-Renditen** je Titel. Log-Renditen addieren sich über die Zeit,
  deshalb wird ein Mehrtageshorizont durch Summieren simulierter Tagespfade gebildet,
  nicht durch Wurzel-Zeit-Skalierung eines Tages-VaR.
* Ausgabe ist die **einfache Portfolio-Rendite** über den Horizont bei Buy-and-hold
  (keine Umschichtung innerhalb des Horizonts): R_p = sum_i w_i * (exp(sum_t r_it) - 1).
  Einfache Renditen addieren sich über Titel, Log-Renditen nicht.
* Die Modelle simulieren **ohne Drift** (Erwartungswert der Log-Rendite = 0). Die Jahre
  2021 bis 2026 hatten für KI-Titel eine außergewöhnlich hohe Durchschnittsrendite; sie in die
  Zukunft fortzuschreiben, würde das Verlustrisiko über 10 Tage künstlich verkleinern.
* Jede Funktion bekommt einen festen `seed`, damit Ergebnisse reproduzierbar sind.
  Große Simulationen laufen in Blöcken (`chunk`), damit der Speicherbedarf klein bleibt.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import special

EWMA_LAMBDA = 0.94  # RiskMetrics-Standard für Tagesdaten


def _as_array(log_returns) -> np.ndarray:
    x = np.asarray(log_returns, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if np.isnan(x).any():
        raise ValueError("Renditen enthalten fehlende Werte")
    return x


def _weights(weights, n_assets: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != n_assets:
        raise ValueError(f"{w.shape[0]} Gewichte für {n_assets} Ticker")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"Gewichte summieren sich zu {w.sum():.4f}, erwartet 1.0")
    return w


def resolve_covariance(log_returns, cov=None) -> np.ndarray:
    """Kovarianzmatrix für die Simulation: die übergebene (z. B. ``RiskDataset.cov``) oder die Stichprobe.

    Wird ``cov`` übergeben, rechnen alle Modelle mit genau der Matrix aus ``build_dataset``.
    Sind Renditen und Matrix beschriftet, wird die Matrix nach den Spalten der Renditen sortiert;
    fehlt ein Ticker in der Matrix, bricht die Funktion ab.
    """
    x = _as_array(log_returns)
    d = x.shape[1]
    if cov is None:
        return np.cov(x, rowvar=False).reshape(d, d)
    if isinstance(cov, pd.DataFrame) and isinstance(log_returns, pd.DataFrame):
        missing = [t for t in log_returns.columns if t not in cov.columns]
        if missing:
            raise ValueError(f"Kovarianzmatrix ohne {missing}")
        cov = cov.loc[log_returns.columns, log_returns.columns]
    c = np.asarray(cov, dtype=float)
    if c.shape != (d, d):
        raise ValueError(f"Kovarianzmatrix hat Form {c.shape}, erwartet ({d}, {d})")
    if not np.allclose(c, c.T):
        raise ValueError("Kovarianzmatrix ist nicht symmetrisch")
    return c


def horizon_portfolio_return(cum_log_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Buy-and-hold-Rendite des Portfolios aus kumulierten Log-Renditen je Titel."""
    return np.expm1(cum_log_returns) @ weights


def _run_chunks(n_sims: int, chunk: int, simulate_chunk: Callable[[int], np.ndarray]) -> np.ndarray:
    out = np.empty(n_sims)
    done = 0
    while done < n_sims:
        size = min(chunk, n_sims - done)
        out[done : done + size] = simulate_chunk(size)
        done += size
    return out


# --------------------------------------------------------------------------- Normalverteilung


def simulate_normal(log_returns, weights, horizon: int = 1, n_sims: int = 1_000_000, seed: int = 42, chunk: int = 100_000, cov=None):
    """Multivariate Normalverteilung mit historischer Kovarianz, Tage unabhängig. Vergleichsmodell zu Schritt 2 v1."""
    x = _as_array(log_returns)
    w = _weights(weights, x.shape[1])
    chol = np.linalg.cholesky(resolve_covariance(log_returns, cov))
    rng = np.random.default_rng(seed)

    def one(size: int) -> np.ndarray:
        cum = np.zeros((size, x.shape[1]))
        for _ in range(horizon):
            cum += rng.standard_normal((size, x.shape[1])) @ chol.T
        return horizon_portfolio_return(cum, w)

    return _run_chunks(n_sims, chunk, one)


# --------------------------------------------------------------------------- Student-t


def multivariate_t_loglik(x: np.ndarray, nu: float, cov=None) -> float:
    """Log-Likelihood einer multivariaten t-Verteilung mit Mittelwert 0 und der Stichprobenkovarianz.

    Die Formmatrix wird so gewählt, dass die Kovarianz der t-Verteilung der Stichprobenkovarianz
    entspricht: Shape = Cov * (nu - 2) / nu. Damit bleibt die Volatilität gleich und nur die
    Form der Ränder (Fat Tails) hängt von nu ab.
    """
    n, d = x.shape
    shape = resolve_covariance(x, cov) * (nu - 2) / nu
    chol = np.linalg.cholesky(shape)
    solved = np.linalg.solve(chol, x.T)
    maha = np.sum(solved**2, axis=0)
    log_det = 2 * np.sum(np.log(np.diag(chol)))
    const = special.gammaln((nu + d) / 2) - special.gammaln(nu / 2) - d / 2 * np.log(nu * np.pi) - 0.5 * log_det
    return float(n * const - (nu + d) / 2 * np.sum(np.log1p(maha / nu)))


def fit_t_degrees_of_freedom(log_returns, grid: np.ndarray | None = None, cov=None) -> float:
    """Schätzt die Freiheitsgrade nu per Maximum-Likelihood auf einem feinen Gitter (2.1 bis 50).

    Kleine nu bedeuten dicke Ränder; nu -> unendlich ist die Normalverteilung.
    """
    c = resolve_covariance(log_returns, cov) if cov is not None else None
    x = _as_array(log_returns)
    x = x - x.mean(axis=0)
    grid = np.arange(2.1, 50.01, 0.05) if grid is None else np.asarray(grid)
    ll = np.array([multivariate_t_loglik(x, nu, c) for nu in grid])
    return float(grid[int(np.argmax(ll))])


def simulate_student_t(
    log_returns,
    weights,
    horizon: int = 1,
    n_sims: int = 1_000_000,
    seed: int = 42,
    nu: float | None = None,
    chunk: int = 100_000,
    cov=None,
):
    """Multivariate Student-t: gleiche Kovarianz wie die Normalverteilung, aber dicke Ränder.

    Ein Tag: Z ~ N(0, Shape), W ~ Chi2(nu), r = Z * sqrt(nu / W). Alle Titel teilen sich
    denselben Mischfaktor W, dadurch fallen an Krisentagen alle gleichzeitig stark (Tail-Abhängigkeit).
    """
    x = _as_array(log_returns)
    w = _weights(weights, x.shape[1])
    c = resolve_covariance(log_returns, cov)
    nu = fit_t_degrees_of_freedom(x, cov=c) if nu is None else float(nu)
    if nu <= 2:
        raise ValueError("nu muss größer als 2 sein, sonst ist die Varianz unendlich")
    d = x.shape[1]
    chol = np.linalg.cholesky(c * (nu - 2) / nu)
    rng = np.random.default_rng(seed)

    def one(size: int) -> np.ndarray:
        cum = np.zeros((size, d))
        for _ in range(horizon):
            z = rng.standard_normal((size, d)) @ chol.T
            mix = np.sqrt(nu / rng.chisquare(nu, size))[:, None]
            cum += z * mix
        return horizon_portfolio_return(cum, w)

    return _run_chunks(n_sims, chunk, one)


# --------------------------------------------------------------------------- Bootstrap


def simulate_block_bootstrap(
    log_returns,
    weights,
    horizon: int = 1,
    n_sims: int = 1_000_000,
    seed: int = 42,
    block: int = 5,
    chunk: int = 100_000,
):
    """Block-Bootstrap: zieht ganze historische Tagesvektoren (alle Titel gleichzeitig).

    Dadurch bleiben die echte Verteilung inklusive Fat Tails und die Korrelation zwischen
    den Titeln erhalten. Für Mehrtageshorizonte werden zusammenhängende Blöcke der Länge
    `block` gezogen (zirkulär), damit Volatilitätscluster innerhalb des Horizonts erhalten bleiben.
    Für einen Tag ist das Verfahren gleich dem einfachen Bootstrap.
    """
    x = _as_array(log_returns)
    x = x - x.mean(axis=0)  # ohne Drift, wie die anderen Modelle
    w = _weights(weights, x.shape[1])
    n = x.shape[0]
    block = max(1, min(block, horizon))
    n_blocks = int(np.ceil(horizon / block))
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)

    def one(size: int) -> np.ndarray:
        starts = rng.integers(0, n, size=(size, n_blocks))
        idx = ((starts[:, :, None] + offsets) % n).reshape(size, -1)[:, :horizon]
        cum = x[idx].sum(axis=1)
        return horizon_portfolio_return(cum, w)

    return _run_chunks(n_sims, chunk, one)


# --------------------------------------------------------------------------- Gefilterte historische Simulation


def ewma_volatility(log_returns, lam: float = EWMA_LAMBDA, warmup: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """EWMA-Volatilität je Titel.

    Gibt (sigma, sigma_next) zurück: sigma[t] ist die Volatilität, die am Vortag für Tag t
    bekannt war; sigma_next ist die Prognose für den Tag nach dem letzten Datenpunkt.
    """
    x = _as_array(log_returns)
    var = np.empty_like(x)
    var[0] = x[:warmup].var(axis=0, ddof=1) if x.shape[0] > warmup else x.var(axis=0, ddof=1)
    for t in range(1, x.shape[0]):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2
    var_next = lam * var[-1] + (1 - lam) * x[-1] ** 2
    return np.sqrt(var), np.sqrt(var_next)


def simulate_filtered_historical(
    log_returns,
    weights,
    horizon: int = 1,
    n_sims: int = 1_000_000,
    seed: int = 42,
    lam: float = EWMA_LAMBDA,
    chunk: int = 100_000,
):
    """Gefilterte historische Simulation (FHS, Barone-Adesi u. a. 1999).

    1. Jede Tagesrendite wird durch ihre EWMA-Volatilität geteilt (standardisierte Residuen).
    2. Gezogen werden ganze Residuenvektoren eines historischen Tages (Korrelation bleibt erhalten).
    3. Die Residuen werden mit der **aktuellen** Volatilität skaliert, und die Volatilität wird
       entlang des simulierten Pfades nach der EWMA-Regel fortgeschrieben.

    Ergebnis: echte Fat Tails aus den Daten, aber auf das heutige Volatilitätsniveau bezogen.
    Ist der Markt gerade ruhig, ist der FHS-VaR kleiner als der historische, in unruhigen Phasen größer.
    """
    x = _as_array(log_returns)
    x = x - x.mean(axis=0)
    w = _weights(weights, x.shape[1])
    sigma, sigma_next = ewma_volatility(x, lam)
    resid = x / sigma
    resid = resid / resid.std(axis=0, ddof=1)  # Einheitsvarianz, damit die Skalierung stimmt
    n, d = resid.shape
    rng = np.random.default_rng(seed)

    def one(size: int) -> np.ndarray:
        var = np.broadcast_to(sigma_next**2, (size, d)).copy()
        cum = np.zeros((size, d))
        for _ in range(horizon):
            r = resid[rng.integers(0, n, size)] * np.sqrt(var)
            cum += r
            var = lam * var + (1 - lam) * r**2
        return horizon_portfolio_return(cum, w)

    return _run_chunks(n_sims, chunk, one)


# --------------------------------------------------------------------------- Historische Simulation


def historical_horizon_returns(log_returns: pd.DataFrame, weights, horizon: int = 1) -> pd.Series:
    """Tatsächliche Buy-and-hold-Portfolio-Renditen über alle (überlappenden) Fenster von `horizon` Tagen.

    Das ist die historische Simulation für Mehrtageshorizonte ohne Wurzel-Zeit-Skalierung.
    Überlappende Fenster sind stark voneinander abhängig: aus 1250 Tagen entstehen bei 10 Tagen
    nur etwa 125 unabhängige Beobachtungen.
    """
    w = _weights(weights, log_returns.shape[1])
    cum = log_returns.rolling(horizon).sum().dropna()
    return pd.Series(horizon_portfolio_return(cum.to_numpy(), w), index=cum.index, name=f"{horizon}d")
