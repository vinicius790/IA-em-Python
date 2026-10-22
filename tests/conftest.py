"""Configuração dos testes.

Tudo que a aplicação grava (banco, modelos, artefatos, catálogo) vai para um
diretório temporário criado aqui, antes de qualquer import de `app`. Os testes
nunca escrevem dentro do repositório.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

RUNTIME_DIR = Path(tempfile.mkdtemp(prefix="iris-ai-tests-"))
os.environ.update(
    {
        "ENVIRONMENT": "development",
        "API_KEY": "",
        "ARTIFACT_SIGNING_KEY": "",
        "MLFLOW_TRACKING_URI": "",
        # `TEST_DATABASE_URL` roda a suíte contra outro banco (ex.: um PostgreSQL vazio, com o
        # driver `psycopg` instalado); sem ela, um SQLite descartável.
        "DATABASE_URL": os.environ.get("TEST_DATABASE_URL")
        or f"sqlite:///{(RUNTIME_DIR / 'test.db').as_posix()}",
        "MODEL_DIR": str(RUNTIME_DIR / "models"),
        "ARTIFACT_DIR": str(RUNTIME_DIR / "artifacts"),
        "DATA_DIR": str(RUNTIME_DIR / "data"),
        "RATE_LIMIT_REQUESTS": "100000",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
VIRGINICA = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2}


@pytest.fixture(scope="session")
def client():
    from app.database import init_db

    init_db()
    with TestClient(app) as test_client:
        yield test_client


def train(client: TestClient, model_name: str) -> dict:
    response = client.post("/api/v1/training/train", json={"model_name": model_name, "cv_folds": 3})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="session")
def trained(client: TestClient) -> dict:
    """Uma regressão logística treinada e registrada, compartilhada pela sessão."""
    return train(client, "logistic_regression")


@pytest.fixture
def api_key(monkeypatch) -> str:
    """Liga a autenticação durante o teste."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "api_key", "chave-de-teste")
    return "chave-de-teste"
