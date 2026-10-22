"""CLI, migrações e consistência do projeto."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from typer.testing import CliRunner

from app import __version__, database
from app.cli import app as cli
from app.config import get_settings
from app.database import Base, engine

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_cli_trains_predicts_and_explains(client):
    listing = runner.invoke(cli, ["models"])
    assert listing.exit_code == 0 and "xgboost" in listing.output

    trained = runner.invoke(cli, ["train", "--model", "knn", "--cv-folds", "3"])
    assert trained.exit_code == 0, trained.output
    assert '"status": "completed"' in trained.output

    predicted = runner.invoke(cli, ["predict", "5.1", "3.5", "1.4", "0.2", "--model", "knn"])
    assert predicted.exit_code == 0, predicted.output
    assert '"predicted_class": "setosa"' in predicted.output

    explained = runner.invoke(cli, ["explain", "5.1", "3.5", "1.4", "0.2", "--model", "knn"])
    assert explained.exit_code == 0 and "exact_shapley" in explained.output


def test_cli_tuning_failure_is_visible(client):
    """Regressão: `--tune` não podia ser ignorado em silêncio."""
    result = runner.invoke(cli, ["train", "--model", "knn", "--cv-folds", "2", "--tune"])
    assert result.exit_code == 1
    assert "no hyperparameter search space" in result.output


def test_cli_batch_score_writes_predictions(client, tmp_path):
    runner.invoke(cli, ["train", "--model", "naive_bayes", "--cv-folds", "3"])
    source = tmp_path / "entrada.csv"
    source.write_text(
        "sepal_length,sepal_width,petal_length,petal_width\n5.1,3.5,1.4,0.2\n6.7,3.0,5.8,2.2\n",
        encoding="utf-8",
    )
    output = tmp_path / "saida.csv"
    result = runner.invoke(
        cli, ["batch-score", str(source), "--model", "naive_bayes", "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith("prediction,confidence")
    assert [line.split(",")[4] for line in lines[1:]] == ["setosa", "virginica"]


def test_cli_module_entry_point_exposes_every_command():
    """Regressão: `if __name__ == "__main__"` ficava antes de pipeline/audit/export-openapi."""
    result = subprocess.run(
        [sys.executable, "-m", "app.cli", "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for command in ("pipeline", "audit", "export-openapi", "batch-score", "monitor", "explain"):
        assert command in result.stdout


def test_cli_exports_openapi(client, tmp_path):
    target = tmp_path / "openapi.json"
    assert runner.invoke(cli, ["export-openapi", "--output", str(target)]).exit_code == 0
    schema = json.loads(target.read_text(encoding="utf-8"))
    assert schema["info"]["version"] == __version__
    assert "/api/v2/explain" in schema["paths"]


def test_migrations_and_orm_describe_the_same_schema(client):
    """Alembic é o único caminho de schema; ele precisa bater com os modelos ORM."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert tables == set(Base.metadata.tables) | {"alembic_version"}
    for name, table in Base.metadata.tables.items():
        assert {column["name"] for column in inspector.get_columns(name)} == set(table.columns.keys()), name
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
            == "0007_canary_attribution"
        )


def test_database_created_by_old_create_all_is_adopted(tmp_path, monkeypatch):
    legacy = create_engine(f"sqlite:///{(tmp_path / 'legado.db').as_posix()}")
    Base.metadata.create_all(legacy)
    with legacy.begin() as connection:
        connection.execute(text("CREATE TABLE model_deployments (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text("INSERT INTO iris_records VALUES (1, 5.1, 3.5, 1.4, 0.2, 'setosa', 'sklearn', '2026-01-01')")
        )
    monkeypatch.setattr(database, "engine", legacy)
    database.init_db()
    tables = set(inspect(legacy).get_table_names())
    assert "alembic_version" in tables and "model_deployments" not in tables
    with legacy.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM iris_records")).scalar() == 1
    legacy.dispose()


def test_runtime_paths_are_outside_the_repository():
    settings = get_settings()
    for path in (settings.model_dir, settings.artifact_dir, settings.data_dir):
        assert not Path(path).resolve().is_relative_to(ROOT), path
    assert str(ROOT) not in settings.database_url.replace("/", os.sep)


def test_version_has_a_single_source():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert not re.search(r"^version\s*=", pyproject, flags=re.MULTILINE)
    assert get_settings().app_version == __version__
    chart = (ROOT / "infra/helm/iris-ai-platform/Chart.yaml").read_text(encoding="utf-8")
    assert f'appVersion: "{__version__}"' in chart
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r"^## (\S+)", changelog, flags=re.MULTILINE).group(1) == __version__
    assert not (ROOT / "VERSION").exists()


def test_console_never_uses_inner_html_and_sends_the_api_key():
    script = (ROOT / "app/console/app.js").read_text(encoding="utf-8")
    code = "\n".join(line for line in script.splitlines() if not line.strip().startswith("//"))
    assert "innerHTML" not in code
    assert 'headers["X-API-Key"] = apiKey' in code
    assert 'api("/api/v4/pipelines/dry-run", { method: "POST"' in code
    page = (ROOT / "app/console/index.html").read_text(encoding="utf-8")
    assert '<html lang="pt-BR">' in page and 'name="viewport"' in page
