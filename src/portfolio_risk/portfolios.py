"""Vorgeschlagene Portfolios für die AI-Bubble-Analyse. Gewichte summieren sich jeweils zu 1."""

TICKERS = ["NVDA", "AVGO", "TSM", "AMD", "MU", "MSFT", "GOOGL", "AMZN", "META", "PLTR"]

PORTFOLIOS = {
    "A_Hauptportfolio": [0.18, 0.10, 0.10, 0.06, 0.06, 0.12, 0.10, 0.10, 0.10, 0.08],
    "B_gleichgewichtet": [0.10] * 10,
    "C_chip_lastig": [0.25, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05],
}

# Anzeigenamen und Gruppe je Ticker (für Dashboard und Schnittstelle)
NAMES = {
    "NVDA": "Nvidia",
    "AVGO": "Broadcom",
    "TSM": "TSMC",
    "AMD": "AMD",
    "MU": "Micron",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "PLTR": "Palantir",
}
GROUPS = {
    **{t: "chips" for t in ("NVDA", "AVGO", "TSM", "AMD", "MU")},
    **{t: "platforms" for t in ("MSFT", "GOOGL", "AMZN", "META", "PLTR")},
}
