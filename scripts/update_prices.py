"""Aktualisiert die Kursdateien in data/raw mit den täglichen Kursen von Yahoo Finance.

    pip install -r requirements-update.txt
    python scripts/update_prices.py             # laden, prüfen, Dateien ersetzen
    python scripts/update_prices.py --dry-run   # nur laden und prüfen

Die Prüfungen stehen in src/portfolio_risk/update.py. Schlägt eine fehl, bleibt der alte Stand
unverändert und das Skript endet mit Fehlercode 1. Läuft das Skript in GitHub Actions, schreibt es
``changed`` und ``last_day`` als Ausgaben des Schritts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from portfolio_risk.portfolios import TICKERS  # noqa: E402
from portfolio_risk.update import UpdateError, update_directory  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="prüfen, aber nichts schreiben")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw", help="Ordner mit den Kurs-CSVs")
    args = parser.parse_args(argv)

    try:
        result = update_directory(args.data_dir, TICKERS, dry_run=args.dry_run)
    except UpdateError as error:
        print(error, file=sys.stderr)
        return 1

    if result.new_days:
        print(f"{len(result.new_days)} neue Handelstage: {result.new_days[0]} bis {result.new_days[-1]}")
    else:
        print(f"Keine neuen Handelstage seit {result.previous_last_day}")
    if result.changed and not result.new_days:
        print("Die Historie hat sich geändert (z. B. Dividendenbereinigung oder mehr Nachkommastellen).")
    for note in result.notes:
        print("Hinweis:", note)
    print("Dateien", "würden ersetzt" if args.dry_run and result.changed else "ersetzt" if result.changed else "unverändert", f"(Stand {result.last_day})")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"changed={'true' if result.changed and not args.dry_run else 'false'}\n")
            handle.write(f"last_day={result.last_day}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
