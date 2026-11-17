# Dienst für das Dashboard: Schnittstelle (FastAPI), Rechenkern und Kursdaten in einem Image.
# --no-access-log: Der Dienst schreibt keine Zeile je Anfrage und damit keine IP-Adressen ins Protokoll.
# Läuft überall, wo Container laufen. Auf Render wird stattdessen render.yaml ohne Docker benutzt.
#
#   docker build -t portfolio-risk-api .
#   docker run --rm -p 8000:8000 portfolio-risk-api
ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PORT=8000
WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src ./src
COPY data/raw ./data/raw

RUN useradd --create-home app
USER app
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn portfolio_risk.api:create_app --factory --host 0.0.0.0 --port ${PORT} --no-access-log"]
