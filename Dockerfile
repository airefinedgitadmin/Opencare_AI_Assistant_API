# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

FROM python:3.12-slim AS runtime

RUN groupadd --system careplan && useradd --system --gid careplan --create-home careplan

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY src ./src

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER careplan
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"]

CMD ["uvicorn", "careplan_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
