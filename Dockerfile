# syntax=docker/dockerfile:1.7

# -----------------------------------------------------------------------------
# Stage 1: Builder - compila dependências e prepara o virtual environment
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md alembic.ini ./
COPY app ./app

RUN pip install --upgrade pip && \
    pip install --no-deps . && \
    pip install .

# -----------------------------------------------------------------------------
# Stage 2: Runtime Minimal - imagem enxuta e segura para produção
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    DATABASE_URL=sqlite:////app/state/data/iris.db \
    DATA_DIR=/app/state/data \
    MODEL_DIR=/app/state/models \
    ARTIFACT_DIR=/app/state/artifacts \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Usuário não-privilegiado (UID/GID fixos para conformidade Helm/Kubernetes runAsUser)
RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --no-create-home app

# Copiar virtualenv pronto do builder
COPY --from=builder /opt/venv /opt/venv

# Copiar arquivos da aplicação e documentação
COPY pyproject.toml README.md alembic.ini ./
COPY app ./app
COPY docs ./docs

# Criar diretórios de estado persistente com permissões corretas
RUN mkdir -p /app/state/data /app/state/models /app/state/artifacts && \
    chown -R app:app /app/state

USER app
VOLUME ["/app/state"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

# Em produção, a inicialização valida API_KEY obrigatória
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
