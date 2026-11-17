"""Schreibt die Startdaten des Dashboards als JavaScript-Datei für die Website.

Die Dashboard-Seite zeigt diesen gespeicherten Stand sofort an und ersetzt ihn, sobald der Dienst
antwortet (``GET /api/snapshot`` liefert denselben Aufbau mit den neuesten Kursen). Ist der Dienst
nicht erreichbar, z. B. weil er gerade startet, bleibt die Seite bedienbar: Wertentwicklung und
Drawdown rechnet sie selbst, VaR und ES der drei vorgegebenen Portfolios stehen in der Datei.
Nur VaR und ES für eigene Gewichte brauchen den Dienst.

    python scripts/build_snapshot.py --out ../Elias-Tandel.github.io/risiko-dashboard-daten.js

Inhalt (``window.RISK_SNAPSHOT``): siehe ``RiskEngine.snapshot`` und dashboard/DATEN.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from portfolio_risk.dataset import build_dataset  # noqa: E402
from portfolio_risk.engine import RiskEngine  # noqa: E402
from portfolio_risk.portfolios import TICKERS  # noqa: E402

HEADER = "/* Gespeicherter Stand des Risiko-Dashboards. Erzeugt mit scripts/build_snapshot.py (portfolio_risk), nicht von Hand ändern. */\n"


def snapshot_json(snapshot: dict) -> str:
    """Kompaktes JSON ohne NaN. Dieselbe Form liefert der Dienst unter /api/snapshot."""
    return json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def write_snapshot(snapshot: dict, path: Path) -> None:
    path.write_text(f"{HEADER}window.RISK_SNAPSHOT={snapshot_json(snapshot)};\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="Zieldatei, z. B. risiko-dashboard-daten.js im Website-Ordner")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args(argv)

    engine = RiskEngine(build_dataset(args.data_dir, TICKERS))
    write_snapshot(engine.snapshot(), args.out)
    print(f"{args.out}: {args.out.stat().st_size / 1024:.0f} KB, Kurse bis {engine.dates[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
