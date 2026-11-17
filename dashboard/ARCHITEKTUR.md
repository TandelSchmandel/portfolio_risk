# Schritt 3: Architektur des Dashboards

Stand: 17.11.2026. Das Dashboard ist eine Unterseite der eigenen Website (`risiko-dashboard.html` im
Repository `Elias-Tandel.github.io`, gehostet bei Hostinger). Besucher wählen Gewichte, Konfidenz und
Horizont; die Seite zeigt VaR, ES, Wertentwicklung und Drawdown auf Deutsch oder Englisch.

## Überblick

```
GitHub Actions (Di–Sa 05:23 UTC)                    Rechendienst (FastAPI, z. B. Render)
  scripts/update_prices.py                            src/portfolio_risk/api.py
  Yahoo Finance → prüfen → data/raw/*.csv ──commit──►   Neustart mit neuen Kursen, Szenarien einmal simulieren
                                                        GET /api/snapshot   Startdaten (gepackt, einmal je Start berechnet)
                                                        GET /api/curve      VaR/ES für eigene Gewichte
                                                                 ▲
                                                                 │ fetch, nur lesend, ohne Cookies
Website bei Hostinger (statisch)                                 │
  risiko-dashboard.html ─────────────────────────────────────────┘
  risiko-dashboard-daten.js   gespeicherter Stand: sofort sichtbar, Ersatz, solange der Dienst schläft
```

## Wer rechnet was

| Größe | Wo | Warum |
|---|---|---|
| Wertentwicklung, Drawdown, Rendite p. a., Volatilität, größter Rückgang | im Browser | Aus den Kursverläufen je Titel (`growth`) und der Kovarianzmatrix ist das ein Skalarprodukt je Tag. Die Diagramme folgen den Reglern ohne Wartezeit und funktionieren auch ohne Dienst. Gleiche Formeln wie `engine.series` und `engine.kpis`; geprüft für 207 Gewichtungen, Abweichung höchstens 0,0055 Punkte (Rundung). |
| VaR und ES der drei vorgegebenen Portfolios | vorab im Dienst, als Raster | 20 Horizonte × 20 Konfidenzstufen je Portfolio stecken im Snapshot. Regler für Konfidenz und Horizont brauchen keine Anfrage. |
| VaR und ES für eigene Gewichte | im Dienst, `GET /api/curve` | Monte-Carlo mit 100 000 Szenarien je Horizont. Eine Anfrage liefert alle 20 Konfidenzstufen für einen Horizont; die Seite fragt erst, wenn ein Regler 120 ms ruht, und merkt sich jede Antwort. |

Der Rechenkern (`engine.py`) simuliert die Szenarien **je Titel** einmal und hält sie im Speicher
(Student-t: 20 Horizonte × 100 000 × 10 Titel als float32, 80 MB). Ein Portfolio ist danach ein
Matrix-Vektor-Produkt plus eine Teilsortierung: etwa 7 ms je Kurve auf einem vollen Rechenkern.
Mit gleicher Simulationszahl und gleichem Seed sind die Werte exakt die aus `models.simulate_*`.
Gegenüber dem Bericht (1 Mio. Simulationen) weicht der VaR nur im Rahmen des Simulationsrauschens ab
(A, 99 %, 10 Tage: 14,57 % statt 14,45 %); die Seite nennt dazu das 95-%-Intervall.

## Verhalten der Seite

1. Die Seite zeichnet sofort den gespeicherten Stand (`window.RISK_SNAPSHOT`).
2. Ist `API_BASE` gesetzt, holt sie `/api/snapshot`. Ein schlafender Dienst braucht bis zu einer Minute;
   die Seite versucht es bis zu 150 s lang (Abstände 2, 4, 8, 15, 30 s) und zeigt solange „Der
   Rechendienst startet“. Danach „nicht erreichbar“ mit Knopf „Erneut versuchen“.
3. Antwortet der Dienst, ersetzt sie den Stand durch den aktuellen (Status „Live berechnet“).
   Meldet eine Kurve ein neueres Datenende als die Seite, lädt sie die Startdaten neu.
4. Bei 429 (zu viele Anfragen) wartet sie 8 s und fragt erneut.

Der Parameter `?api=http://localhost:8000` (nur `localhost` oder `127.0.0.1`) verbindet die Seite zum
Testen mit einem lokalen Dienst. Andere Adressen werden ignoriert, damit niemand der Seite über einen
Link fremde Zahlen unterschieben kann.

## Schutz und Datenschutz

* Der Dienst ist öffentlich und nur lesend. Er nimmt außer Gewichten, Konfidenz und Horizont keine
  Eingaben an, prüft sie (Fehler → 422) und begrenzt Rechen-Endpunkte auf 300 Anfragen je Minute und
  Besucher (429). Die Zählung liegt nur im Arbeitsspeicher und wird nach einer Minute verworfen.
* Der Dienst schreibt keine Zeile je Anfrage ins Protokoll (`--no-access-log`).
* Die Seite sendet ihre Anfragen ohne Cookies und ohne Referrer und lädt nichts von fremden Servern.
* CORS steht auf `*`, solange die Website unter mehreren Adressen erreichbar ist. Einschränken:
  `RISK_ALLOWED_ORIGINS=https://elias-tandel.de,https://www.elias-tandel.de` (siehe [`BETRIEB.md`](BETRIEB.md)).

## Dateien

| Datei | Inhalt |
|---|---|
| `src/portfolio_risk/engine.py` | Rechenkern: Szenarien im Speicher, `portfolio`, `curve`, `snapshot` |
| `src/portfolio_risk/api.py` | HTTP-Schnittstelle, Prüfung der Eingaben, Begrenzung, gepackter Snapshot |
| `src/portfolio_risk/update.py`, `scripts/update_prices.py` | tägliche Kurse laden und prüfen |
| `.github/workflows/update-data.yml` | Zeitplan für die Aktualisierung, Tests, Commit |
| `render.yaml`, `Dockerfile`, `.python-version`, `requirements-api.txt` | Betrieb des Dienstes |
| `scripts/build_snapshot.py` | gespeicherten Stand für die Website schreiben |
| `scripts/export_dashboard.py`, `dashboard/data/*.json` | Export der Berichtsergebnisse aus Schritt 2 (Schema v1), für spätere Ansichten wie Backtest und Stresstests |

Aufbau der Antworten: [`DATEN.md`](DATEN.md). Einrichten und Betrieb: [`BETRIEB.md`](BETRIEB.md).

## Mögliche nächste Ansichten

Die erste Fassung zeigt Wertentwicklung und Drawdown. Die Daten für weitere Ansichten liegen bereits vor:
VaR-Modelle im Vergleich (`/api/portfolio?models=all`), Backtest mit Ampel und Stresstests
(`dashboard/data/kennzahlen.json`, `zeitreihen.json`), Korrelationen (`kennzahlen.json`).
