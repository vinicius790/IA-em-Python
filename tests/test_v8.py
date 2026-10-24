"""8.1 de ponta a ponta: datasets enviados, comparação de modelos, canário e usuários com papéis."""

import io
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select
from sqlalchemy import text as sql_text
from sqlalchemy.exc import IntegrityError
from typer.testing import CliRunner

from app import database
from app.cli import app as cli
from app.config import get_settings
from app.data import load_iris_dataframe
from app.database import SessionLocal
from app.db_models import ApiUser, CanaryDeployment, PredictionLog
from app.governance.rbac import hash_key
from app.ml import train_model
from app.modeling.comparison import compare_models
from app.quality import validate_training_frame
from tests.conftest import SETOSA

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()
CSV = {"Content-Type": "text/csv"}


def wine_csv(rows: int = 60) -> str:
    """Dataset sintético separável: 3 features numéricas, 1 categórica e o alvo `tipo`."""
    lines = ["acidez,acucar,teor,regiao,tipo"]
    for i in range(rows):
        red = i % 2 == 1
        acidity = (3.0 if red else 7.0) + (i % 10) * 0.1
        sugar = (10.0 if red else 2.0) + (i // 2 % 7) * 0.3
        alcohol = 9.0 + (i * 7 % 13) * 0.25
        region, kind = ("sul" if i % 3 else "norte"), ("tinto" if red else "branco")
        lines.append(f"{acidity:.2f},{sugar:.2f},{alcohol:.2f},{region},{kind}")
    return "\n".join(lines) + "\n"


RED = {"acidez": 3.2, "acucar": 10.5, "teor": 10.0}
WHITE = {"acidez": 7.4, "acucar": 2.3, "teor": 11.0}


def upload(client, name: str, text: str | None = None, **params):
    return client.post(
        "/api/v8/datasets/upload", params={"name": name, **params}, content=text or wine_csv(), headers=CSV
    )


@pytest.fixture(scope="session")
def wines(client) -> dict:
    response = upload(client, "vinhos")
    assert response.status_code == 201, response.text
    return response.json()


def train_on(client, dataset: str, model: str) -> dict:
    response = client.post(
        "/api/v1/training/train", json={"model_name": model, "dataset": dataset, "cv_folds": 3}
    )
    assert response.status_code == 200, response.text
    return response.json()


def version_of(client, name: str) -> int:
    return client.get(f"/api/v1/models/{name}").json()["version"]


# ------------------------------------------------------------------ datasets


def test_upload_profiles_infers_the_target_and_registers_in_the_catalog(client, wines):
    assert wines["target"] == "tipo" and wines["target_inferred"] is True
    assert wines["features"] == ["acidez", "acucar", "teor"]
    assert wines["classes"] == {"branco": 30, "tinto": 30}
    assert wines["rows"] == 60 and len(wines["sha256"]) == 64
    columns = {column["name"]: column for column in wines["profile"]["column_profiles"]}
    assert columns["acidez"]["semantic_type"] == "numeric"
    assert columns["regiao"]["semantic_type"] == "categorical"
    stored = Path(get_settings().data_dir) / "uploads" / "vinhos" / f"{wines['sha256'][:16]}.csv"
    assert stored.read_text(encoding="utf-8") == wine_csv()

    listed = client.get("/api/v8/datasets").json()
    assert [row["name"] for row in listed].count("vinhos") == 1
    assert "profile" not in listed[0]
    single = client.get("/api/v8/datasets/vinhos").json()
    assert single["sha256"] == wines["sha256"]
    # O caminho do arquivo no servidor fica só no catálogo interno, nunca na resposta.
    assert all("path" not in body for body in (wines, single, *listed))
    assert all("path" not in body for body in client.get("/api/v3/datasets").json())
    assert client.get("/api/v8/datasets/nao-existe").status_code == 404
    actions = [event["action"] for event in client.get("/api/v4/audit").json()]
    assert "dataset_upload" in actions


def test_profile_route_does_not_store_anything(client):
    response = client.post("/api/v8/datasets/profile", content="a,b\n1,x\n2,y\n", headers=CSV)
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == 2 and body["training_problems"]
    assert not (Path(get_settings().data_dir) / "uploads" / "a.csv").exists()


def test_invalid_uploads_are_422_with_the_reason(client):
    for name, text, params, reason in [
        ("../fora", wine_csv(), {}, "Dataset name"),
        ("iris", wine_csv(), {}, "reserved"),
        ("pequeno", wine_csv(10), {}, "30 rows"),
        ("sem-alvo", wine_csv(), {"target": "inexistente"}, "not found"),
        ("quebrado", "a,b\n1,2,3\n", {}, "line 2"),
        ("vazio", "\n", {}, "empty"),
    ]:
        response = upload(client, name, text, **params)
        assert response.status_code == 422, (name, response.text)
        assert reason in response.json()["detail"], name
    binary = client.post(
        "/api/v8/datasets/upload", params={"name": "bin"}, content=b"\xff\xfe\x00", headers=CSV
    )
    assert binary.status_code == 422 and "UTF-8" in binary.json()["detail"]
    assert not list((Path(get_settings().data_dir) / "uploads").glob("pequeno*"))


def test_explicit_target_and_reupload_register_a_new_version(client):
    first = upload(client, "vinhos-regiao", target="regiao").json()
    assert first["target"] == "regiao" and first["target_inferred"] is False
    second = upload(client, "vinhos-regiao", wine_csv(90), target="regiao").json()
    assert second["rows"] == 90 and second["sha256"] != first["sha256"]
    assert client.get("/api/v8/datasets/vinhos-regiao").json()["sha256"] == second["sha256"]


def test_models_keep_the_dataset_version_they_were_trained_on(client):
    """Reenviar o nome com outro schema não pode quebrar o drift nem a robustez dos modelos antigos."""
    assert upload(client, "versionado").status_code == 201
    first = train_on(client, "versionado", "naive_bayes")
    name = first["model_name"]
    renamed = wine_csv().replace("acidez,acucar,teor", "ph,brix,graduacao")
    assert upload(client, "versionado", renamed).json()["features"] == ["ph", "brix", "graduacao"]

    record = client.get(f"/api/v1/models/{name}").json()
    old_files = list((Path(get_settings().data_dir) / "uploads" / "versionado").glob("*.csv"))
    assert len(old_files) == 2 and record["metadata"]["dataset_sha256"]
    for i in range(30):
        features = {"acidez": 3.0 + i % 10 * 0.1, "acucar": 10.0 + i % 7 * 0.3, "teor": 9.5}
        assert (
            client.post("/api/v8/predict", json={"model_name": name, "features": features}).status_code == 200
        )
    summary = client.get(f"/api/v2/monitoring/{name}")
    assert summary.status_code == 200
    assert set(summary.json()["drift"]["features"]) == {"acidez", "acucar", "teor"}
    robustness = client.post("/api/v6/modeling/robustness", params={"model_name": name})
    assert robustness.status_code == 200, robustness.text
    # Um modelo novo treina na versão mais recente.
    assert train_on(client, "versionado", "knn")["metrics"]["classes"] == ["branco", "tinto"]
    newer = client.get("/api/v1/models/versionado.knn").json()["metadata"]["dataset_sha256"]
    assert newer != record["metadata"]["dataset_sha256"]


def test_uploaded_values_are_read_for_training_exactly_as_profiled(client):
    """Cabeçalho e valores com espaços e números com `_` são lidos pelo treino com as mesmas
    regras do perfil (com o `read_csv` do pandas o treino falhava ou mudava as classes)."""
    lines = wine_csv().splitlines()
    body = [f"{line.replace(',tinto', ', tinto ')},1_0" for line in lines[1:]]
    text = "\n".join([" acidez , acucar,teor,regiao, tipo ,codigo", *body]) + "\n"
    response = upload(client, "divergente", text, target="tipo")
    assert response.status_code == 201, response.text
    uploaded = response.json()
    assert uploaded["features"] == ["acidez", "acucar", "teor"]  # `1_0` não é número
    assert uploaded["classes"] == {"branco": 30, "tinto": 30}
    run = train_on(client, "divergente", "logistic_regression")
    assert run["metrics"]["classes"] == ["branco", "tinto"]


def test_uploads_that_would_fail_the_training_gates_are_rejected(client):
    with_nulls = wine_csv().replace("\n3.10,", "\n,", 1)
    assert with_nulls != wine_csv()
    missing = upload(client, "nulos", with_nulls)
    assert missing.status_code == 422 and "feature columns have missing values" in missing.json()["detail"]
    repeated = wine_csv() + "\n".join(wine_csv().splitlines()[1:4]) + "\n"
    duplicated = upload(client, "repetido", repeated)
    assert duplicated.status_code == 422 and "repeat the same features" in duplicated.json()["detail"]
    huge_field = "a,b\n" + "x" * 200_000 + ",1\n"
    malformed = upload(client, "campo-enorme", huge_field)
    assert malformed.status_code == 422 and "Malformed CSV" in malformed.json()["detail"]


def test_training_on_an_uploaded_dataset_registers_a_dataset_scoped_model(client, wines):
    run = train_on(client, "vinhos", "logistic_regression")
    assert run["model_name"] == "vinhos.logistic_regression" and run["status"] == "completed"
    assert run["metrics"]["classes"] == ["branco", "tinto"]
    assert run["metrics"]["accuracy"] == 1.0
    record = client.get("/api/v1/models/vinhos.logistic_regression").json()
    assert record["metadata"]["dataset"] == "vinhos"
    assert [item["feature"] for item in record["metrics"]["feature_importance"]] != []
    assert {item["feature"] for item in record["metrics"]["feature_importance"]} == set(wines["features"])

    unknown = client.post("/api/v1/training/train", json={"model_name": "knn", "dataset": "nao-existe"})
    assert unknown.status_code == 422 and "Unknown dataset" in unknown.json()["detail"]
    iris = client.post("/api/v1/training/train", json={"model_name": "knn", "dataset": "iris", "cv_folds": 3})
    assert iris.json()["model_name"] == "knn"


def test_named_prediction_and_explanation_for_a_dataset_model(client, wines):
    train_on(client, "vinhos", "logistic_regression")
    name = "vinhos.logistic_regression"
    red = client.post("/api/v8/predict", json={"model_name": name, "features": RED}).json()
    white = client.post("/api/v8/predict", json={"model_name": name, "features": WHITE}).json()
    assert (red["predicted_class"], white["predicted_class"]) == ("tinto", "branco")
    assert red["arm"] == "stable" and red["canary_id"] is None and red["prediction_id"] > 0

    wrong = client.post("/api/v8/predict", json={"model_name": name, "features": {"acidez": 1.0, "x": 2.0}})
    assert wrong.status_code == 422 and "missing" in wrong.json()["detail"]
    # A rota v1 é a do Iris (4 features): contra um modelo de 3 features responde 422, não 500.
    iris_shape = client.post("/api/v1/predict", json={**SETOSA, "model_name": name})
    assert iris_shape.status_code == 422 and "expects 3 features" in iris_shape.json()["detail"]
    assert client.post("/api/v2/explain", json={**SETOSA, "model_name": name}).status_code == 422
    untrained = client.post("/api/v8/predict", json={"model_name": "vinhos.svm", "features": RED})
    assert untrained.status_code == 404
    # `NaN`/`Infinity` (JSON não padrão que o parser aceita) são 422 com o motivo. Antes, na v8,
    # chegavam ao modelo; e em qualquer rota o handler padrão ecoava o valor e a resposta virava 500.
    json_body = {"Content-Type": "application/json"}
    for value in ("NaN", "Infinity"):
        raw = f'{{"model_name": "{name}", "features": {{"acidez": {value}, "acucar": 1, "teor": 1}}}}'
        for route in ("/api/v8/predict", "/api/v8/explain"):
            invalid = client.post(route, content=raw, headers=json_body)
            assert invalid.status_code == 422, (route, value, invalid.text)
            assert invalid.json()["detail"][0]["type"] == "finite_number"
        iris_raw = f'{{"sepal_length": {value}, "sepal_width": 3, "petal_length": 1, "petal_width": 0.2}}'
        assert client.post("/api/v1/predict", content=iris_raw, headers=json_body).status_code == 422

    response = client.post("/api/v8/explain", json={"model_name": name, "features": RED})
    explanation = response.json()["explanation"]
    assert explanation["predicted_class"] == "tinto"
    assert {item["feature"] for item in explanation["contributions"]} == set(wines["features"])
    total = sum(item["shap_value"] for item in explanation["contributions"])
    assert total == pytest.approx(explanation["prediction"] - explanation["base_value"], abs=1e-6)


def test_monitoring_uses_the_dataset_the_model_was_trained_on(client, wines):
    train_on(client, "vinhos", "naive_bayes")
    name = "vinhos.naive_bayes"
    for i in range(32):
        features = {"acidez": 3.0 + i % 10 * 0.1, "acucar": 10.0 + i % 7 * 0.3, "teor": 9.0 + i % 13 * 0.25}
        response = client.post("/api/v8/predict", json={"model_name": name, "features": features})
        assert response.status_code == 200
    summary = client.get(f"/api/v2/monitoring/{name}").json()
    assert summary["predictions"] == 32
    assert set(summary["drift"]["features"]) == set(wines["features"])
    assert not any(p["reason"] == "dataset_quality_failed" for p in summary["health"]["penalties"])


def test_quality_gates_and_training_accept_custom_columns(tmp_path):
    import pandas as pd

    frame = pd.read_csv(io.StringIO(wine_csv()))
    features = ["acidez", "acucar", "teor"]
    result = validate_training_frame(frame, features, "tipo")
    assert result.passed
    assert "positive_measurements" not in {check["name"] for check in result.checks}
    missing = validate_training_frame(frame, [*features, "ph"], "tipo")
    assert [c["observed"] for c in missing.checks if c["name"] == "required_columns"] == [["ph"]]
    # Sem argumentos continua sendo o gate do Iris, com a regra de medidas positivas.
    iris_checks = validate_training_frame(load_iris_dataframe()).checks
    assert "positive_measurements" in {check["name"] for check in iris_checks}

    trained = train_model(
        frame, "knn", tmp_path, cv_folds=3, features=features, target="tipo", dataset="vinhos"
    )
    assert trained.model_name == "vinhos.knn"
    assert Path(trained.artifact_path).name.startswith("vinhos.knn_")


# ---------------------------------------------------------------- comparison


def test_comparison_ranks_models_with_confidence_intervals_and_writes_reports(client):
    response = client.post(
        "/api/v8/comparisons",
        json={
            "models": ["naive_bayes", "logistic_regression", "decision_tree"],
            "folds": 4,
            "confidence": 0.99,
        },
    )
    assert response.status_code == 201, response.text
    report = response.json()
    assert report["dataset"] == "iris" and report["protocol"]["folds"] == 4
    assert report["protocol"]["confidence"] == 0.99 and report["protocol"]["classes"] == 3
    results = report["results"]
    assert [row["rank"] for row in results] == [1, 2, 3]
    assert report["best_model"] == results[0]["model"]
    means = [row["f1_macro"]["mean"] for row in results]
    assert means == sorted(means, reverse=True)
    for row in results:
        for metric in ("f1_macro", "accuracy"):
            interval = row[metric]
            assert interval["lower"] <= interval["mean"] <= interval["upper"]
            assert interval["n"] == 4 and len(row["fold_scores"][metric]) == 4
            assert interval["mean"] == pytest.approx(sum(row["fold_scores"][metric]) / 4, abs=1e-5)
    best = results[0]
    assert best["f1_macro_delta_to_best"]["mean"] == 0 and best["differs_from_best"] is False

    run_id = report["run_id"]
    assert Path(report["reports"]["json"]).is_file() and Path(report["reports"]["html"]).is_file()
    assert client.get(f"/api/v8/comparisons/{run_id}").json()["results"] == results
    page = client.get(f"/api/v8/comparisons/{run_id}/html")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert run_id in page.text and best["model"] in page.text
    assert client.get("/api/v8/comparisons").json()[0]["run_id"] == run_id
    assert client.get("/api/v8/comparisons/..%2F..%2Fsegredo").status_code == 404
    assert client.get("/api/v8/comparisons/20260101T000000-deadbeef").status_code == 404
    assert "model_comparison" in [event["action"] for event in client.get("/api/v4/audit").json()]


def test_comparison_on_an_uploaded_dataset_and_invalid_requests(client, wines):
    response = client.post(
        "/api/v8/comparisons", json={"models": ["knn", "naive_bayes"], "dataset": "vinhos", "folds": 3}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["dataset"] == "vinhos" and body["protocol"]["target"] == "tipo"
    assert body["protocol"]["features"] == wines["features"]

    for payload, reason in [
        ({"models": ["knn", "nao_existe"]}, "Unknown models"),
        ({"models": ["knn", "knn"]}, "two distinct"),
        ({"models": ["knn", "naive_bayes"], "dataset": "sumiu"}, "Unknown dataset"),
    ]:
        invalid = client.post("/api/v8/comparisons", json=payload)
        assert invalid.status_code == 422 and reason in invalid.json()["detail"]
    assert client.post("/api/v8/comparisons", json={"models": ["knn"]}).status_code == 422
    low = client.post("/api/v8/comparisons", json={"models": ["knn", "svm"], "confidence": 0.5})
    assert low.status_code == 422


def test_compare_models_uses_identical_folds_and_is_reproducible():
    frame = load_iris_dataframe()
    features = [column for column in frame.columns if column != "class_name"]
    first = compare_models(frame, features, "class_name", ["knn", "naive_bayes"], folds=3, random_state=7)
    second = compare_models(frame, features, "class_name", ["naive_bayes", "knn"], folds=3, random_state=7)
    assert {r["model"]: r["fold_scores"] for r in first["results"]} == {
        r["model"]: r["fold_scores"] for r in second["results"]
    }
    with pytest.raises(ValueError, match="smallest class"):
        compare_models(frame.groupby("class_name").head(3), features, "class_name", ["knn", "svm"], folds=5)


# -------------------------------------------------------------------- canary


def two_versions(client, model: str) -> tuple[str, int, int]:
    """Treina duas versões do modelo no dataset de vinhos e põe a primeira em produção."""
    name = f"vinhos.{model}"
    train_on(client, "vinhos", model)
    stable = version_of(client, name)
    train_on(client, "vinhos", model)
    candidate = version_of(client, name)
    promoted = client.post(
        f"/api/v2/models/{name}/promote", json={"version": stable, "stage": "production", "force": True}
    )
    assert promoted.status_code == 200, promoted.text
    return name, stable, candidate


def test_canary_splits_traffic_keeps_separate_metrics_and_promotes_by_rule(client, wines, monkeypatch):
    monkeypatch.setattr(get_settings(), "gate_max_latency_ms", 60_000.0)
    name, stable, candidate = two_versions(client, "decision_tree")
    started = client.post(
        "/api/v8/canary",
        json={
            "model_name": name,
            "candidate_version": candidate,
            "traffic_percent": 50,
            # A latência de dois modelos iguais varia só por ruído; a razão larga evita teste instável.
            "rule": {"min_requests": 5, "max_latency_ratio": 100},
        },
    )
    assert started.status_code == 201, started.text
    canary = started.json()
    assert canary["status"] == "active" and canary["evaluation"]["decision"] == "wait"
    assert canary["arms"]["stable"]["version"] == stable
    assert canary["arms"]["candidate"]["version"] == candidate

    arms, sent = [], 0
    for i in range(200):
        body = client.post(
            "/api/v8/predict", json={"model_name": name, "features": RED, "routing_key": f"usuario-{i}"}
        ).json()
        sent += 1
        arms.append(body["arm"])
        assert body["canary_id"] == canary["id"]
        assert body["model_version"] == (candidate if body["arm"] == "candidate" else stable)
        if body["canary_status"] == "promoted":
            break
    assert arms.count("candidate") == 5 and "stable" in arms, "promove na 5ª predição da candidata"

    report = client.get(f"/api/v8/canary/{canary['id']}").json()
    assert report["status"] == "promoted" and report["decided_at"]
    assert report["decision"]["decision"] == "promote" and report["decision"]["automatic"] is True
    assert report["arms"]["candidate"]["requests"] == 5
    assert report["arms"]["stable"]["requests"] == sent - 5
    record = client.get(f"/api/v1/models/{name}").json()
    assert (record["version"], record["stage"]) == (candidate, "production")
    # A rota v1 continuou servindo a estável durante o canário; agora todos recebem a nova versão.
    after = client.post("/api/v8/predict", json={"model_name": name, "features": RED}).json()
    assert (after["arm"], after["canary_id"], after["model_version"]) == ("stable", None, candidate)
    events = {event["action"] for event in client.get("/api/v4/audit", params={"limit": 500}).json()}
    assert {"canary_started", "canary_promoted"} <= events
    assert client.post(f"/api/v8/canary/{canary['id']}/evaluate").status_code == 409


def test_canary_rolls_back_by_rule_and_archives_the_candidate(client, wines):
    name, stable, candidate = two_versions(client, "extra_trees")
    canary = client.post(
        "/api/v8/canary",
        json={"model_name": name, "candidate_version": candidate, "rule": {"min_requests": 5}},
    ).json()
    assert canary["traffic_percent"] == 10
    duplicate = client.post("/api/v8/canary", json={"model_name": name, "candidate_version": candidate})
    assert duplicate.status_code == 409 and "already has an active canary" in duplicate.json()["detail"]
    assert client.post(f"/api/v8/canary/{canary['id']}/evaluate").json()["status"] == "active"

    with SessionLocal() as db:
        for version, confidence, count in [(stable, 0.95, 10), (candidate, 0.55, 5)]:
            db.add_all(
                PredictionLog(
                    model_name=name,
                    model_version=version,
                    predicted_class="tinto",
                    confidence=confidence,
                    features=[3.2, 10.5, 10.0],
                    latency_ms=1.0,
                )
                for _ in range(count)
            )
        db.commit()

    report = client.post(f"/api/v8/canary/{canary['id']}/evaluate").json()
    assert report["status"] == "rolled_back"
    assert report["decision"]["reason"] == "candidate failed: confidence"
    assert report["arms"]["stable"]["mean_confidence"] == 0.95
    assert report["arms"]["candidate"]["mean_confidence"] == 0.55
    models = client.get("/api/v1/models").json()
    stages = {row["version"]: row["stage"] for row in models if row["name"] == name}
    assert stages[candidate] == "archived" and stages[stable] == "production"
    assert client.post(f"/api/v8/canary/{canary['id']}/rollback").status_code == 409
    archived = client.post("/api/v8/canary", json={"model_name": name, "candidate_version": candidate})
    assert archived.status_code == 409
    listing = client.get("/api/v8/canary", params={"model_name": name}).json()
    assert [(row["id"], row["status"]) for row in listing] == [(canary["id"], "rolled_back")]


def test_canary_manual_rollback_traffic_change_and_conflicts(client, wines):
    name, stable, candidate = two_versions(client, "knn")
    missing = client.post("/api/v8/canary", json={"model_name": name, "candidate_version": 999})
    assert missing.status_code == 404
    same = client.post("/api/v8/canary", json={"model_name": name, "candidate_version": stable})
    assert same.status_code == 409
    train_on(client, "vinhos", "svm")
    no_production = client.post("/api/v8/canary", json={"model_name": "vinhos.svm", "candidate_version": 1})
    assert no_production.status_code == 409 and "no production version" in no_production.json()["detail"]
    assert client.get("/api/v8/canary/999999").status_code == 404
    zero = {"model_name": name, "candidate_version": candidate, "traffic_percent": 0}
    assert client.post("/api/v8/canary", json=zero).status_code == 422
    beyond_window = {"model_name": name, "candidate_version": candidate, "rule": {"min_requests": 501}}
    assert client.post("/api/v8/canary", json=beyond_window).status_code == 422

    canary = client.post("/api/v8/canary", json={"model_name": name, "candidate_version": candidate}).json()
    changed = client.post(f"/api/v8/canary/{canary['id']}/traffic", json={"traffic_percent": 100})
    assert changed.json()["traffic_percent"] == 100
    routed = client.post("/api/v8/predict", json={"model_name": name, "features": WHITE}).json()
    assert routed["arm"] == "candidate" and routed["canary_status"] == "active"

    rolled = client.post(f"/api/v8/canary/{canary['id']}/rollback").json()
    assert rolled["status"] == "rolled_back" and rolled["decision"]["reason"] == "manual rollback"
    late = client.post(f"/api/v8/canary/{canary['id']}/traffic", json={"traffic_percent": 5})
    assert late.status_code == 409
    back = client.post("/api/v8/predict", json={"model_name": name, "features": WHITE}).json()
    assert (back["arm"], back["model_version"]) == ("stable", stable)


def test_a_canary_is_decided_only_once_under_concurrency(client, wines):
    """Duas decisões sobre o mesmo canário (ex.: duas predições simultâneas da candidata, ou
    uma reversão manual durante a promoção automática): só a primeira mexe no registry."""
    from app.errors import CanaryConflictError
    from app.services import get_service

    service = get_service()
    name, stable, candidate = two_versions(client, "linear_discriminant")
    canary_id = client.post(
        "/api/v8/canary", json={"model_name": name, "candidate_version": candidate}
    ).json()["id"]
    with SessionLocal() as first, SessionLocal() as second:
        mine, stale = first.get(CanaryDeployment, canary_id), second.get(CanaryDeployment, canary_id)
        service.rollback_canary(first, mine, {"decision": "rollback", "reason": "primeira"})
        assert stale.status == "active"  # a outra sessão ainda vê o canário ativo
        with pytest.raises(CanaryConflictError, match="already rolled_back"):
            service.rollback_canary(second, stale, {"decision": "rollback", "reason": "segunda"})
        assert stale.status == "rolled_back" and stale.decision["reason"] == "primeira"
    events = client.get("/api/v4/audit", params={"limit": 500}).json()
    closed = [
        e for e in events if e["action"] == "canary_rolled_back" and e["payload"]["canary_id"] == canary_id
    ]
    assert len(closed) == 1
    stages = {
        row["version"]: row["stage"] for row in client.get("/api/v1/models").json() if row["name"] == name
    }
    assert (stages[stable], stages[candidate]) == ("production", "archived")


# --------------------------------------------------------------------- users


def create_user(client, master: dict, name: str, role: str) -> dict:
    response = client.post("/api/v8/users", json={"name": name, "role": role}, headers=master)
    assert response.status_code == 201, response.text
    return response.json()


def test_roles_limit_what_each_user_key_can_do(client, trained, api_key):
    master = {"X-API-Key": api_key}
    assert client.get("/api/v8/users/me", headers=master).json() == {"name": "api-key", "role": "admin"}
    reader = {"X-API-Key": create_user(client, master, "leitora", "reader")["api_key"]}
    operator = {"X-API-Key": create_user(client, master, "operador", "operator")["api_key"]}
    admin = {"X-API-Key": create_user(client, master, "gestora", "admin")["api_key"]}

    assert client.get("/api/v8/users/me", headers=reader).json() == {"name": "leitora", "role": "reader"}
    assert client.get("/api/v1/models", headers=reader).status_code == 200
    prediction = client.post(
        "/api/v1/predict", json={**SETOSA, "model_name": "logistic_regression"}, headers=reader
    )
    assert prediction.status_code == 200
    denied = client.post("/api/v1/training/train", json={"model_name": "knn"}, headers=reader)
    assert denied.status_code == 403 and "requires 'operator'" in denied.json()["detail"]
    assert upload(client, "do-leitor").status_code == 401
    as_reader = client.post(
        "/api/v8/datasets/upload", params={"name": "x"}, content="a\n", headers=reader | CSV
    )
    assert as_reader.status_code == 403

    assert client.post("/api/v1/dataset/seed", headers=operator).status_code == 200
    assert client.get("/api/v8/users", headers=operator).status_code == 403
    forbidden = client.post("/api/v8/users", json={"name": "intruso", "role": "admin"}, headers=operator)
    assert forbidden.status_code == 403 and "requires 'admin'" in forbidden.json()["detail"]

    listed = client.get("/api/v8/users", headers=admin).json()
    assert {"leitora", "operador", "gestora"} <= {user["name"] for user in listed}
    assert all("api_key" not in user and "key_hash" not in user for user in listed)
    assert client.get("/api/v1/models", headers={"X-API-Key": "iris_chave-inventada"}).status_code == 401


def test_user_keys_are_stored_hashed_revocable_and_audited(client, api_key):
    master = {"X-API-Key": api_key}
    created = create_user(client, master, "auditada", "operator")
    key, headers = created["api_key"], {"X-API-Key": created["api_key"]}
    assert key.startswith("iris_")
    with SessionLocal() as db:
        stored = db.scalar(select(ApiUser).where(ApiUser.name == "auditada"))
        assert stored.key_hash == hash_key(key) and key not in stored.key_hash

    assert client.post("/api/v1/dataset/seed", headers=headers).status_code == 200
    events = client.get("/api/v4/audit", headers=master).json()
    seed = next(event for event in events if event["action"] == "dataset_seed")
    assert seed["actor"] == "auditada" and seed["payload"]["actor_role"] == "operator"
    creation = next(event for event in events if event["action"] == "user_created")
    assert creation["actor"] == "api-key" and creation["payload"]["user"] == "auditada"
    assert key not in str(events)

    again = client.post("/api/v8/users", json={"name": "auditada", "role": "reader"}, headers=master)
    assert again.status_code == 409
    for invalid in (
        {"name": "api-key", "role": "admin"},
        {"name": "com espaço", "role": "reader"},
        {"name": "x", "role": "root"},
    ):
        assert client.post("/api/v8/users", json=invalid, headers=master).status_code == 422

    revoked = client.delete(f"/api/v8/users/{created['id']}", headers=master)
    assert revoked.status_code == 200 and revoked.json()["active"] is False
    assert client.get("/api/v1/models", headers=headers).status_code == 401
    assert client.delete("/api/v8/users/999999", headers=master).status_code == 404
    assert "user_revoked" in [event["action"] for event in client.get("/api/v4/audit", headers=master).json()]


def test_open_development_mode_acts_as_admin(client):
    assert client.get("/api/v8/users/me").json() == {"name": "anonymous", "role": "admin"}
    assert client.get("/api/v8/users").status_code == 200


# ----------------------------------------------------------------------- CLI


def test_cli_adds_a_dataset_trains_on_it_and_compares_models(client, tmp_path):
    source = tmp_path / "vinhos.csv"
    source.write_text(wine_csv(), encoding="utf-8")
    added = runner.invoke(cli, ["dataset-add", str(source), "--name", "cli-vinhos"])
    assert added.exit_code == 0, added.output
    assert '"target": "tipo"' in added.output
    assert "cli-vinhos" in runner.invoke(cli, ["datasets"]).output

    rejected = runner.invoke(cli, ["dataset-add", str(source), "--name", "Nome Inválido"])
    assert rejected.exit_code == 1 and "Dataset name" in rejected.output

    trained = runner.invoke(
        cli, ["train", "--model", "naive_bayes", "--dataset", "cli-vinhos", "--cv-folds", "3"]
    )
    assert trained.exit_code == 0, trained.output
    assert '"model_name": "cli-vinhos.naive_bayes"' in trained.output

    before = set((Path(get_settings().artifact_dir) / "comparisons").glob("*.html"))
    compared = runner.invoke(
        cli, ["compare", "naive_bayes", "knn", "--dataset", "cli-vinhos", "--folds", "3"]
    )
    assert compared.exit_code == 0, compared.output
    assert "naive_bayes" in compared.output
    assert len(set((Path(get_settings().artifact_dir) / "comparisons").glob("*.html")) - before) == 1
    assert runner.invoke(cli, ["compare", "knn", "nao_existe"]).exit_code == 1


def test_cli_manages_users_and_canaries(client, wines):
    created = runner.invoke(cli, ["user-create", "pela-cli", "--role", "operator"])
    assert created.exit_code == 0 and '"api_key": "iris_' in created.output
    assert runner.invoke(cli, ["user-create", "pela-cli"]).exit_code == 1
    assert runner.invoke(cli, ["user-create", "outra", "--role", "root"]).exit_code == 1
    assert "pela-cli" in runner.invoke(cli, ["users"]).output

    name, _, candidate = two_versions(client, "random_forest")
    started = runner.invoke(
        cli, ["canary-start", name, str(candidate), "--traffic", "25", "--min-requests", "3"]
    )
    assert started.exit_code == 0, started.output
    assert '"status": "active"' in started.output and '"created_by": "cli"' in started.output
    assert runner.invoke(cli, ["canary-start", name, str(candidate)]).exit_code == 1
    canary_id = client.get("/api/v8/canary", params={"model_name": name}).json()[0]["id"]
    status = runner.invoke(cli, ["canary-status", str(canary_id), "--evaluate"])
    assert status.exit_code == 0 and '"decision": "wait"' in status.output
    assert runner.invoke(cli, ["canary-status", "999999"]).exit_code == 1
    assert client.post(f"/api/v8/canary/{canary_id}/rollback").status_code == 200


# ----------------------------------------------------------------- migration


def test_migration_0005_can_be_downgraded_and_reapplied(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'migracao.db').as_posix()}")
    monkeypatch.setattr(database, "engine", engine)
    database.init_db()

    def schema() -> tuple[set[str], set[str]]:
        inspector = inspect(engine)
        return set(inspector.get_table_names()), {c["name"] for c in inspector.get_columns("prediction_logs")}

    tables, columns = schema()
    assert {"api_users", "canary_deployments"} <= tables and "model_version" in columns

    def active_index() -> bool:
        return "uq_canary_deployments_active_model" in {
            index["name"] for index in inspect(engine).get_indexes("canary_deployments")
        }

    config = Config()
    config.set_main_option("script_location", str(database.MIGRATIONS_DIR))
    assert active_index()
    command.downgrade(config, "0005_users_canary")
    assert not active_index()
    # Um banco em que a corrida já criou dois ativos: a 0006 fecha o mais antigo e cria o índice.
    with engine.begin() as connection:
        for _ in range(2):
            connection.execute(
                sql_text(
                    "INSERT INTO canary_deployments (model_name, stable_version, candidate_version, "
                    "traffic_percent, status, rule, created_at) "
                    "VALUES ('m', 1, 2, 10, 'active', '{}', CURRENT_TIMESTAMP)"
                )
            )
    command.upgrade(config, "head")
    assert active_index()
    with engine.connect() as connection:
        rows = connection.execute(sql_text("SELECT status FROM canary_deployments ORDER BY id")).all()
    assert [row[0] for row in rows] == ["rolled_back", "active"]
    command.downgrade(config, "0004_drop_model_deployments")
    tables, columns = schema()
    assert not {"api_users", "canary_deployments"} & tables and "model_version" not in columns

    command.upgrade(config, "head")
    tables, columns = schema()
    assert {"api_users", "canary_deployments"} <= tables and "model_version" in columns
    engine.dispose()


def test_migrations_0005_and_0006_generate_offline_sql():
    """`alembic upgrade/downgrade --sql` não tem banco para inspecionar nem para o batch refletir."""
    sql = io.StringIO()
    config = Config(output_buffer=sql)
    config.set_main_option("script_location", str(database.MIGRATIONS_DIR))
    command.upgrade(config, "0004_drop_model_deployments:0005_users_canary", sql=True)
    assert "CREATE TABLE api_users" in sql.getvalue()
    assert "ALTER TABLE prediction_logs ADD COLUMN model_version" in sql.getvalue()
    command.downgrade(config, "0005_users_canary:0004_drop_model_deployments", sql=True)
    assert "ALTER TABLE prediction_logs DROP COLUMN model_version" in sql.getvalue()
    assert "DROP TABLE api_users" in sql.getvalue()

    sql.seek(0)
    sql.truncate()
    command.upgrade(config, "0005_users_canary:0006_unique_active_canary", sql=True)
    created = " ".join(sql.getvalue().split())
    assert (
        "CREATE UNIQUE INDEX uq_canary_deployments_active_model ON canary_deployments (model_name) "
        "WHERE status = 'active'"
    ) in created
    command.downgrade(config, "0006_unique_active_canary:0005_users_canary", sql=True)
    assert "DROP INDEX uq_canary_deployments_active_model" in sql.getvalue()


# ------------------------------------------------- concorrência e limites (8.1)


def test_only_one_active_canary_per_model_even_under_a_race(client, wines):
    """A checagem da aplicação e o INSERT não são atômicos; o índice único parcial é quem garante."""
    from app.deployment.canary import CanaryRule
    from app.errors import CanaryConflictError
    from app.services import get_service

    service = get_service()
    name, stable, candidate = two_versions(client, "quadratic_discriminant")
    # Duas aberturas simultâneas, cada uma na sua sessão: exatamente uma vence.
    barrier, outcomes = threading.Barrier(2), []

    def open_canary() -> None:
        with SessionLocal() as db:
            barrier.wait()
            try:
                outcomes.append(service.start_canary(db, name, candidate, 10, CanaryRule()).id)
            except CanaryConflictError as exc:
                outcomes.append(str(exc))

    threads = [threading.Thread(target=open_canary) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    winners = [item for item in outcomes if isinstance(item, int)]
    assert len(winners) == 1, outcomes
    assert any("already has an active canary" in str(item) for item in outcomes)

    # Mesmo que a checagem da aplicação deixe passar (a corrida), a API responde 409, não 500.
    original = service.active_canary
    service.active_canary = lambda db, model_name: None
    try:
        raced = client.post("/api/v8/canary", json={"model_name": name, "candidate_version": candidate})
    finally:
        del service.active_canary
    assert service.active_canary == original
    assert raced.status_code == 409 and "already has an active canary" in raced.json()["detail"]

    def canary_row(status: str) -> CanaryDeployment:
        return CanaryDeployment(
            model_name=name,
            stable_version=stable,
            candidate_version=candidate,
            traffic_percent=5,
            status=status,
            rule={},
        )

    with SessionLocal() as db:
        db.add(canary_row("active"))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
        # Encerrados não contam: o histórico pode ter vários por modelo.
        db.add_all([canary_row("rolled_back"), canary_row("promoted")])
        db.commit()
    listing = client.get("/api/v8/canary", params={"model_name": name}).json()
    assert [row["status"] for row in listing].count("active") == 1
    assert client.post(f"/api/v8/canary/{winners[0]}/rollback").status_code == 200


def test_the_canary_rule_runs_once_when_the_candidate_reaches_min_requests(client, wines, monkeypatch):
    """Antes de `min_requests` a regra só pode esperar: cada predição faz uma contagem limitada,
    e o relatório completo roda uma vez, quando o mínimo é atingido (e decide)."""
    from app.services import get_service

    monkeypatch.setattr(get_settings(), "gate_max_latency_ms", 60_000.0)
    service = get_service()
    name, _, candidate = two_versions(client, "linear_sgd")
    rule = {"min_requests": 5, "max_latency_ratio": 100}
    payload = {"model_name": name, "candidate_version": candidate, "traffic_percent": 100, "rule": rule}
    canary = client.post("/api/v8/canary", json=payload).json()
    calls = []
    original = service.evaluate_canary

    def spy(db, current, *, automatic=False):
        calls.append(automatic)
        return original(db, current, automatic=automatic)

    monkeypatch.setattr(service, "evaluate_canary", spy)
    statuses = []
    for i in range(5):
        body = client.post(
            "/api/v8/predict", json={"model_name": name, "features": RED, "routing_key": f"c{i}"}
        ).json()
        statuses.append(body["canary_status"])
    assert statuses == ["active"] * 4 + ["promoted"]
    assert calls == [True]
    report = client.get(f"/api/v8/canary/{canary['id']}").json()
    assert report["status"] == "promoted" and report["arms"]["candidate"]["requests"] == 5


CATALOG_WRITER = """
import pathlib, sys, time
from app.data_catalog.catalog import DatasetCatalog
root, tag, go = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
catalog = DatasetCatalog(root)
while not go.exists():
    time.sleep(0.005)
for i in range(15):
    catalog.register(f"{tag}-{i}", root / "fonte.csv")
"""


def test_catalog_writes_from_two_processes_do_not_lose_records(tmp_path):
    """Ler-alterar-gravar do `catalog.json` sob trava de arquivo: dois processos (dois workers
    do uvicorn, ou a API e a CLI) registrando ao mesmo tempo não perdem registros."""
    (tmp_path / "fonte.csv").write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
    go = tmp_path / "go"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    writers = [
        subprocess.Popen(
            [sys.executable, "-c", CATALOG_WRITER, str(tmp_path), tag, str(go)], cwd=ROOT, env=env
        )
        for tag in ("a", "b")
    ]
    go.touch()
    assert [writer.wait(timeout=180) for writer in writers] == [0, 0]
    from app.data_catalog.catalog import DatasetCatalog

    names = {record["name"] for record in DatasetCatalog(tmp_path).list()}
    assert names == {f"{tag}-{i}" for tag in ("a", "b") for i in range(15)}
    assert not (tmp_path / "catalog.lock").exists()


def test_heavy_cross_validation_routes_have_a_work_budget(client):
    """Comparação, matriz de experimentos e AutoML treinam dentro da requisição: no máximo 60 treinos."""
    too_many = [
        ("/api/v8/comparisons", {"models": ["knn", "svm", "naive_bayes", "decision_tree"], "folds": 16}),
        ("/api/v6/research/experiment-matrix", {"folds": 3, "repeats": 2}),  # todos os modelos
        ("/api/v6/research/experiment-matrix", {"models": ["knn", "svm"], "folds": 10, "repeats": 4}),
        ("/api/v7/labs/automl", {"max_models": 20, "folds": 4}),
    ]
    for route, body in too_many:
        response = client.post(route, json=body)
        assert response.status_code == 422, (route, response.text)
        assert "cross-validation fits" in str(response.json()["detail"]), route
    at_limit = {"models": ["knn", "naive_bayes"], "folds": 20}  # 2 x 20 = 40
    assert client.post("/api/v8/comparisons", json=at_limit).status_code == 201
