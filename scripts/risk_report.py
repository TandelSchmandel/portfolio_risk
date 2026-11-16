"""Erstellt den Risikobericht für alle Portfolios und speichert ihn als CSV."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from portfolio_risk import load_prices, simple_returns  # noqa: E402
from portfolio_risk.portfolios import PORTFOLIOS, TICKERS  # noqa: E402
from portfolio_risk.risk import (  # noqa: E402
    expected_shortfall,
    historical_var,
    monte_carlo_var,
    replay_return,
    uniform_shock,
    worst_window,
)


def main() -> None:
    prices = load_prices(ROOT / "data" / "raw")
    prices = prices[TICKERS]
    returns = simple_returns(prices)

    rows = []
    for name, weights in PORTFOLIOS.items():
        portfolio = returns[TICKERS].to_numpy() @ pd.Series(weights, index=TICKERS).to_numpy()
        portfolio = pd.Series(portfolio, index=returns.index)
        worst_20d, worst_end = worst_window(returns[TICKERS], weights, days=20)
        rows.append(
            {
                "Portfolio": name,
                "VaR 95 % historisch": historical_var(portfolio, 0.95),
                "VaR 99 % historisch": historical_var(portfolio, 0.99),
                "VaR 99 % Monte-Carlo": monte_carlo_var(returns[TICKERS], weights, 0.99),
                "ES 99 % historisch": expected_shortfall(portfolio, 0.99),
                "Stress: Tech-Korrektur 2022 (Jan-Okt)": replay_return(returns[TICKERS], weights, "2022-01-03", "2022-10-14"),
                "Stress: schlechtestes 20-Tage-Fenster": worst_20d,
                "Stress: Ende des Fensters": worst_end.date().isoformat(),
                "Stress: alle Titel -20 %": uniform_shock(weights, -0.20),
            }
        )

    report = pd.DataFrame(rows).set_index("Portfolio")
    out = ROOT / "reports" / "risk_summary.csv"
    report.to_csv(out, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(report.T.to_string())
    print(f"\nGespeichert: {out}")


if __name__ == "__main__":
    main()
