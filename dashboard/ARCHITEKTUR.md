# Schritt 3: Vorschlag für die Dashboard-Architektur

Stand: Vorbereitung. Die offenen Punkte unten hängen von der eigenen Website ab und sind noch nicht entschieden.

## Ausgangslage

* Alle Ergebnisse aus Schritt 2 liegen als CSV in `reports/v2/` und jetzt zusätzlich als JSON in `dashboard/data/`
  (Schema: [`DATEN.md`](DATEN.md)). Zusammen etwa 350 KB, gzip etwa 110 KB.
* Die Rechnung ist teuer (etwa 75 s für 1 Mio. Simulationen je Modell), die Daten ändern sich nur, wenn neue
  Kurse hinzukommen. Für eine Portfolio-Website reicht deshalb ein **vorberechneter, statischer Datenstand**.

## Empfohlene Variante: statisch, Daten vorberechnet

```
data/raw/*.csv ──► risk_report_v2.py ──► reports/v2/*.csv ──► export_dashboard.py ──► dashboard/data/*.json
                                                                                          │
                                                       Website (beliebiges Framework) ◄───┘  fetch() oder Import beim Build
```

* Kein Server, kein Python auf dem Hosting, beliebig günstig (GitHub Pages, Netlify, Vercel, eigener Webspace).
* Das Dashboard ist eine Seite oder Komponente, die die drei JSON-Dateien lädt und Charts zeichnet.
* Aktualisierung: lokal beide Skripte laufen lassen und die JSON-Dateien committen, oder später eine
  GitHub Action, die das z. B. wöchentlich macht (Kursdaten müssten dann automatisch geladen werden).

## Alternativen

| Variante | Vorteil | Nachteil |
|---|---|---|
| **Statisch + JSON** (empfohlen) | einfach, schnell, kostenlos zu hosten, passt in jede Website | keine Live-Neuberechnung |
| Statisch + kleine Rechnung im Browser | Besucher können Gewichte selbst ändern und sehen Volatilität / parametrischen VaR sofort (Kovarianzmatrix ist im Export) | volle Monte-Carlo-Modelle nur näherungsweise |
| Python-Backend (FastAPI) | eigene Portfolios mit allen Modellen live rechnen | Server nötig, Kosten, Laufzeit pro Anfrage, Missbrauchsschutz |
| Streamlit / Dash | sehr schnell gebaut, alles in Python | eigene App neben der Website, schwer ins Design einzubetten |

## Mögliche Ansichten

1. **Überblick**: Kennzahl-Kacheln je Portfolio (VaR 99 % 1T / 10T Student-t, ES, Volatilität, max. Drawdown).
2. **Wertentwicklung und Drawdown**: Linien für A, B, C (`zeitreihen.json`).
3. **VaR-Modelle im Vergleich**: Balken mit Konfidenzintervall je Modell, umschaltbar 95/99 % und 1/10 Tage.
4. **Backtest**: tägliche Verluste mit VaR-Linie, Ausnahmen markiert, Ampel und Kupiec-Ergebnis.
5. **Stresstests**: 21 Szenarien in drei Gruppen, A/B/C nebeneinander.
6. **Portfolio-Zusammensetzung und Korrelation**: Gewichte, Heatmap der Korrelationen.
7. **Methodik**: Kurzfassung von `reports/v2/METHODIK.md` mit Grenzen der Analyse.

## Offene Entscheidungen

Tech-Stack der Website, Einbindung (Unterseite, eingebettete Komponente oder iframe), Hosting, Design
(an die Website angepasst oder eigenständig), Chart-Bibliothek, Interaktivität, Sprache (Deutsch, Englisch
oder beides), Umfang der ersten Version und wie oft die Daten aktualisiert werden.
