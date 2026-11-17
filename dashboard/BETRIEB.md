# Betrieb: Rechendienst einrichten, Daten aktualisieren

Überblick über das Zusammenspiel: [`ARCHITEKTUR.md`](ARCHITEKTUR.md).

## 1. Rechendienst bei Render einrichten (einmalig, etwa 10 Minuten)

Voraussetzung: Dieser Branch ist in `main` gemergt.

1. Bei [render.com](https://render.com) mit dem GitHub-Konto anmelden und Render Zugriff auf das
   Repository `portfolio_risk` geben.
2. **New → Blueprint** wählen, das Repository auswählen. Render liest `render.yaml` und schlägt den Dienst
   `portfolio-risk-api` vor (Python, kostenlose Stufe, Region Frankfurt). **Apply** bestätigen.
3. Der erste Build dauert einige Minuten. Danach zeigt Render die Adresse des Dienstes, etwa
   `https://portfolio-risk-api.onrender.com` (ist der Name vergeben, hängt Render eine Kennung an).
4. Prüfen: `<Adresse>/api/health` antwortet mit `"status": "ok"` und dem Datenstand,
   `<Adresse>/docs` zeigt alle Endpunkte zum Ausprobieren.
5. Auf der Website in `risiko-dashboard.html` die Adresse eintragen (ohne Schrägstrich am Ende) und die
   Seite zu Hostinger hochladen:

   ```js
   let API_BASE = 'https://portfolio-risk-api.onrender.com';
   ```

Kostenlose Stufe: 512 MB Speicher, 0,1 CPU. Nach 15 Minuten ohne Anfrage schläft der Dienst ein; der
nächste Besucher wartet bis zu etwa einer Minute, bis er wieder läuft. Die Seite zeigt in der Zeit den
gespeicherten Stand und meldet „Der Rechendienst startet“. Wer das nicht möchte, wählt in Render eine
kostenpflichtige Stufe; dann läuft der Dienst durchgehend.

### Umgebungsvariablen

| Variable | Standard | Bedeutung |
|---|---|---|
| `RISK_ALLOWED_ORIGINS` | `*` | Herkunftsseiten, die den Dienst aus dem Browser abfragen dürfen, durch Komma getrennt. Sobald die Website-Adresse feststeht, einschränken: `https://elias-tandel.de,https://www.elias-tandel.de` (in Render: Service → Environment). |
| `RISK_RATE_LIMIT` | `300` | Anfragen je Minute und Besucher für `/api/curve`, `/api/portfolio`, `/api/overview`; `0` schaltet die Grenze ab |
| `RISK_N_SIMS` | `100000` | Simulationen je Modell und Horizont. Mehr braucht mehr Speicher (Student-t: 0,8 KB je Simulation) |
| `RISK_DATA_DIR` | `data/raw` | Ordner mit den Kurs-CSVs |

### Anderer Anbieter

`Dockerfile` baut denselben Dienst als Container (Port aus `PORT`, Standard 8000):

```bash
docker build -t portfolio-risk-api .
docker run --rm -p 8000:8000 portfolio-risk-api
```

Der Dienst braucht etwa 200 MB Speicher. Bei einem anderen Anbieter den Abschnitt „Risiko-Dashboard“ in
`datenschutz.html` der Website anpassen.

## 2. Tägliche Aktualisierung der Kurse

`.github/workflows/update-data.yml` läuft Dienstag bis Samstag um 05:23 UTC, nachdem die US-Börse am
Vorabend geschlossen hat:

1. `scripts/update_prices.py` lädt die Kurse aller zehn Titel von Yahoo Finance neu und prüft sie
   (kein fehlender Tag, Tagesrenditen passen zum alten Stand, alle Titel enden am selben Tag,
   Datenprüfung aus `quality.py`). Schlägt eine Prüfung fehl, bleibt der alte Stand, und der Lauf erscheint
   auf GitHub als fehlgeschlagen (je nach Benachrichtigungseinstellung auch per E-Mail).
2. Mit neuen Kursen laufen alle Tests, dann committet der Lauf `data/raw/*.csv` auf `main`.
3. Render baut den Dienst bei diesem Commit neu (`autoDeployTrigger: commit`, nur bei Änderungen in
   `src/`, `data/raw/` und den Abhängigkeiten). Danach liefert er den neuen Stand.

Von Hand starten: auf GitHub unter **Actions → Kursdaten aktualisieren → Run workflow**. Lokal:

```bash
pip install -r requirements.txt -r requirements-update.txt
python scripts/update_prices.py --dry-run   # nur laden und prüfen
python scripts/update_prices.py
```

Hinweis: Der Abruf bei Yahoo Finance konnte aus der Entwicklungsumgebung nicht getestet werden (kein
Zugang). Nach dem Mergen den Lauf einmal von Hand starten und das Protokoll ansehen.

Ist `main` durch Regeln geschützt, die direkte Pushes verbieten, kann der Lauf nicht committen. Dann
entweder eine Ausnahme für GitHub Actions einrichten oder den Lauf auf Pull Requests umstellen.

## 3. Gespeicherten Stand der Website erneuern

Die Website enthält mit `risiko-dashboard-daten.js` einen gespeicherten Stand. Er wird sofort gezeigt
und gilt, solange der Dienst schläft oder nicht erreichbar ist. Er altert nicht schlimm, aber alle paar
Wochen lohnt es sich, ihn zu erneuern (etwa 10 Sekunden):

```bash
python scripts/build_snapshot.py --out ../Elias-Tandel.github.io/risiko-dashboard-daten.js
```

Dann die Datei zu Hostinger hochladen. Ändert sich der Aufbau der Daten (`SCHEMA_VERSION` in
`engine.py`), müssen Seite, Datei und Dienst zusammen erneuert werden; die Seite prüft die Fassung und
ignoriert unpassende Daten.

## 4. Lokal entwickeln

```bash
pip install -r requirements.txt -r requirements-api.txt
pytest -q
uvicorn portfolio_risk.api:create_app --factory --app-dir src --port 8000
```

Im Ordner der Website `python3 -m http.server 8080` starten und
`http://localhost:8080/risiko-dashboard.html?api=http://localhost:8000` öffnen.

## Wenn etwas nicht stimmt

| Beobachtung | Ursache und Abhilfe |
|---|---|
| Seite meldet dauerhaft „nicht erreichbar“ | In Render das Protokoll des Dienstes ansehen. `API_BASE` richtig geschrieben, mit `https://`? Bei eingeschränkten `RISK_ALLOWED_ORIGINS`: steht die Adresse der Website darin? |
| Datenstand bleibt stehen | Unter GitHub Actions den letzten Lauf von „Kursdaten aktualisieren“ ansehen. Eine fehlgeschlagene Prüfung nennt Titel und Tag. |
| Zahlen auf der Seite weichen vom Bericht ab | Erwartet im Rahmen des Simulationsrauschens: Der Dienst rechnet mit 100 000, der Bericht mit 1 Mio. Simulationen, und die Kurse sind neuer. Die Seite nennt das 95-%-Intervall des VaR. |
