"""Fluxos da API de ponta a ponta, com um modelo realmente treinado."""

import time
from pathlib import Path

import pytest

from app.config import get_settings
from app.data import FEATURE_COLUMNS, load_iris_dataframe
from app.data_catalog.catalog import DatasetCatalog
from tests.conftest import SETOSA, VIRGINICA, train


def test_root_and_health_report_the_package_version(client):
    from app import __version__

    assert client.get("/").json()["version"] == __version__
    assert client.get("/healthz").json() == {"status": "ok", "version": __version__}
    assert client.get("/api/v1/health").json()["environment"] == "development"
    assert client.get("/api/v2/platform").json()["version"] == __version__


def test_readiness_needs_a_default_model(client):
    assert client.get("/readyz").json() == {"status": "ready", "database": True}
    body = client.get("/api/v1/ready").json()
    assert body["database"] is True
    assert body["status"] == ("ready" if body["model_available"] else "not_ready")


def test_dashboard_redirects_to_console(client):
    response = client.get("/api/v1/dashboard", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/console/"


def test_dataset_summary_and_seed(client):
    body = client.get("/api/v1/dataset/summary").json()
    assert body == {
        "rows": 150,
        "features": FEATURE_COLUMNS,
        "classes": {"setosa": 50, "versicolor": 50, "virginica": 50},
    }
    first = client.post("/api/v1/dataset/seed", params={"replace": True}).json()
    assert first["inserted"] == 150
    assert client.post("/api/v1/dataset/seed").json()["inserted"] == 0
    assert len(client.get("/api/v1/iris", params={"class_name": "setosa", "limit": 500}).json()) == 50


def test_iris_crud(client):
    created = client.post("/api/v1/iris", json={**SETOSA, "class_name": "setosa"})
    assert created.status_code == 201
    iris_id = created.json()["id"]
    updated = client.put(f"/api/v1/iris/{iris_id}", json={**VIRGINICA, "class_name": "virginica"})
    assert updated.json()["class_name"] == "virginica"
    assert client.get(f"/api/v1/iris/{iris_id}").json()["petal_length"] == 5.8
    assert client.delete(f"/api/v1/iris/{iris_id}").status_code == 200
    assert client.get(f"/api/v1/iris/{iris_id}").status_code == 404


def test_training_registers_a_verifiable_artifact(client, trained):
    assert trained["status"] == "completed"
    record = client.get("/api/v1/models/logistic_regression").json()
    assert record["metrics"]["accuracy"] == trained["metrics"]["accuracy"]
    artifact = Path(get_settings().model_dir) / record["artifact_path"]
    assert artifact.is_file()
    runs = client.get("/api/v1/training/runs").json()
    assert any(run["id"] == trained["run_id"] and run["status"] == "completed" for run in runs)
    experiments = client.get("/api/v2/experiments").json()
    assert any(item["name"] == f"logistic_regression-{trained['run_id']}" for item in experiments)


def test_unknown_model_is_422_and_untrained_model_is_404(client):
    assert client.post("/api/v1/training/train", json={"model_name": "nao_existe"}).status_code == 422
    assert client.post("/api/v2/training/async", json={"model_name": "nao_existe"}).status_code == 422
    missing = client.post("/api/v1/predict", json={**SETOSA, "model_name": "extra_trees"})
    assert missing.status_code == 404
    assert client.post("/api/v2/explain", json={**SETOSA, "model_name": "extra_trees"}).status_code == 404


def test_tuning_an_untunable_model_reports_the_reason(client):
    response = client.post(
        "/api/v1/training/train", json={"model_name": "knn", "cv_folds": 2, "tune_hyperparameters": True}
    )
    assert response.status_code == 422
    assert "no hyperparameter search space" in response.json()["detail"]


def test_prediction_returns_class_version_and_id(client, trained):
    body = client.post("/api/v1/predict", json={**SETOSA, "model_name": "logistic_regression"}).json()
    assert body["predicted_class"] == "setosa"
    assert body["model_version"] >= 1
    assert body["prediction_id"] > 0
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert body["confidence"] == max(body["probabilities"].values())


def test_batch_prediction_keeps_order_across_models_and_loads_each_model_once(client, trained, monkeypatch):
    """Regressão: o batch da v1 fazia uma predição (e uma carga de artefato) por item."""
    train(client, "naive_bayes")
    from app import services
    from app.services import prediction_cache

    prediction_cache.clear()

    calls = []
    original = services.predict_artifact
    monkeypatch.setattr(
        services, "predict_artifact", lambda *args: calls.append(len(args[1])) or original(*args)
    )
    items = [
        {**SETOSA, "model_name": "logistic_regression"},
        {**VIRGINICA, "model_name": "naive_bayes"},
        {**VIRGINICA, "model_name": "logistic_regression"},
        {**SETOSA, "model_name": "naive_bayes"},
    ]
    response = client.post("/api/v1/predict/batch", json={"items": items})
    assert response.status_code == 200
    body = response.json()
    assert [row["model_name"] for row in body] == [item["model_name"] for item in items]
    assert [row["predicted_class"] for row in body] == ["setosa", "virginica", "virginica", "setosa"]
    assert sorted(calls) == [2, 2], "uma predição vetorizada por modelo"
    assert len({row["prediction_id"] for row in body}) == 4


def test_batch_size_limit(client, trained, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_batch_size", 2)
    items = [{**SETOSA, "model_name": "logistic_regression"}] * 3
    assert client.post("/api/v1/predict/batch", json={"items": items}).status_code == 413


def test_explanation_depends_on_the_instance(client, trained):
    """Regressão: a explicação ignorava a instância e devolvia sempre a importância global."""
    setosa = client.post("/api/v2/explain", json={**SETOSA, "model_name": "logistic_regression"}).json()
    virginica = client.post("/api/v2/explain", json={**VIRGINICA, "model_name": "logistic_regression"}).json()
    first, second = setosa["explanation"], virginica["explanation"]
    assert first["method"] == "exact_shapley"
    assert (first["predicted_class"], second["predicted_class"]) == ("setosa", "virginica")
    assert first["contributions"] != second["contributions"]
    for explanation in (first, second):
        total = sum(item["shap_value"] for item in explanation["contributions"])
        assert total == pytest.approx(explanation["prediction"] - explanation["base_value"], abs=1e-5)
        assert {item["feature"] for item in explanation["contributions"]} == set(FEATURE_COLUMNS)
    assert first["global_importance"] == trained["metrics"]["feature_importance"]


def test_tampered_artifact_is_never_loaded(client):
    train(client, "decision_tree")
    record = client.get("/api/v1/models/decision_tree").json()
    artifact = Path(get_settings().model_dir) / record["artifact_path"]
    artifact.write_bytes(artifact.read_bytes() + b"adulterado")
    payload = {**SETOSA, "model_name": "decision_tree"}
    for path in ("/api/v1/predict", "/api/v2/explain"):
        response = client.post(path, json=payload)
        assert response.status_code == 500, path
        assert "integrity" in response.json()["detail"]


def test_monitoring_uses_real_predictions_and_feedback(client):
    train(client, "linear_discriminant")
    assert client.get("/api/v2/monitoring/linear_discriminant").json()["status"] == "no_data"

    frame = load_iris_dataframe()
    items = [
        {**row, "model_name": "linear_discriminant"}
        for row in frame[FEATURE_COLUMNS].to_dict(orient="records")
    ]
    predictions = client.post("/api/v1/predict/batch", json={"items": items}).json()
    for prediction, actual in list(zip(predictions, frame["class_name"], strict=True))[::10]:
        feedback = client.post(
            f"/api/v2/predictions/{prediction['prediction_id']}/feedback", json={"actual_class": actual}
        )
        assert feedback.status_code == 200
        assert feedback.json()["correct"] == (prediction["predicted_class"] == actual)

    summary = client.get("/api/v2/monitoring/linear_discriminant").json()
    confidences = [p["confidence"] for p in predictions]
    assert summary["predictions"] == 150
    assert summary["labeled"] == 15
    assert summary["observed"]["mean_confidence"] == pytest.approx(sum(confidences) / 150, abs=1e-5)
    assert summary["observed"]["accuracy_when_labeled"] >= 0.9
    assert sum(summary["class_distribution"].values()) == 150
    assert summary["drift"]["status"] == "stable", "o próprio dataset de referência não pode acusar drift"
    assert summary["health"] == {"score": 100.0, "status": "healthy", "penalties": []}
    assert (
        client.post("/api/v2/predictions/99999999/feedback", json={"actual_class": "setosa"}).status_code
        == 404
    )


def test_drift_is_detected_and_lowers_health(client):
    train(client, "quadratic_discriminant")
    few = [{**SETOSA, "model_name": "quadratic_discriminant"}] * 5
    client.post("/api/v1/predict/batch", json={"items": few})
    insufficient = client.get("/api/v2/monitoring/drift/quadratic_discriminant").json()
    assert insufficient["status"] == "insufficient_data"
    assert insufficient["report"]["required"] == 30

    shifted = [
        {
            "sepal_length": 7.9,
            "sepal_width": 4.4,
            "petal_length": 6.9,
            "petal_width": 2.5,
            "model_name": "quadratic_discriminant",
        }
    ] * 60
    client.post("/api/v1/predict/batch", json={"items": shifted})
    drift = client.get("/api/v2/monitoring/drift/quadratic_discriminant").json()
    assert drift["status"] == "drift"
    assert set(drift["report"]["drifted_features"]) == set(FEATURE_COLUMNS)

    report = client.post("/api/v2/monitoring/quadratic_discriminant/reports")
    assert report.status_code == 201
    payload = report.json()["payload"]
    assert {"reason": "drift_drift", "points": 35} in payload["health"]["penalties"]
    assert payload["health"]["score"] <= 65
    history = client.get("/api/v2/monitoring/quadratic_discriminant/reports").json()
    assert history[0]["id"] == report.json()["id"]
    metrics = client.get("/api/v1/metrics").text
    assert 'iris_model_drift_psi_max{model="quadratic_discriminant"}' in metrics


def test_promotion_to_production_enforces_the_release_gate(client, trained, monkeypatch):
    settings = get_settings()
    version = client.get("/api/v1/models/logistic_regression").json()["version"]
    payload = {"version": version, "stage": "production", "notes": "teste"}

    monkeypatch.setattr(settings, "gate_min_accuracy", 0.9999)
    blocked = client.post("/api/v2/models/logistic_regression/promote", json=payload)
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["gate"]["passed"] is False
    assert client.get("/api/v1/models/logistic_regression").json()["stage"] == "candidate"
    gate = client.get("/api/v4/governance/gate/logistic_regression").json()
    assert gate["gate"]["passed"] is False

    monkeypatch.setattr(settings, "gate_min_accuracy", 0.5)
    promoted = client.post("/api/v2/models/logistic_regression/promote", json=payload)
    assert promoted.status_code == 200
    assert promoted.json()["stage"] == "production"
    assert (
        client.post(
            "/api/v2/models/logistic_regression/promote", json={**payload, "version": 999}
        ).status_code
        == 404
    )
    actions = [event["action"] for event in client.get("/api/v4/audit").json()]
    assert "promotion" in actions and "promotion_blocked" in actions


def test_async_training_job_is_persisted_and_completes(client):
    response = client.post("/api/v2/training/async", json={"model_name": "naive_bayes", "cv_folds": 3})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(300):
        job = client.get(f"/api/v2/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert job["status"] == "completed", job
    assert job["started_at"] and job["finished_at"]
    assert job["result"]["model_name"] == "naive_bayes"
    assert any(item["id"] == job_id for item in client.get("/api/v2/jobs").json())
    assert client.get("/api/v2/jobs/nao-existe").status_code == 404


def test_data_endpoints_describe_the_real_dataset(client):
    quality = client.get("/api/v2/data/quality").json()
    assert quality["passed"] is True and quality["score"] == 1.0
    profile = client.get("/api/v3/data/profile").json()
    assert profile["rows"] == 150
    assert {column["name"] for column in profile["column_profiles"]} == {*FEATURE_COLUMNS, "class_name"}
    raw = client.get("/api/v2/features/manifest").json()
    engineered = client.get("/api/v2/features/manifest", params={"engineered": True}).json()
    assert len(raw["features"]) == 5
    assert len(engineered["features"]) == 10
    assert client.get("/api/v4/data/anomalies").json()["count"] >= 0
    selection = client.get("/api/v4/features/selection", params={"k": 3}).json()
    assert len(selection["best"]["selected"]) == 3
    assert client.get("/api/v4/features/selection", params={"method": "invalido"}).status_code == 422
    matrix = client.get("/api/v4/analytics/correlations").json()["matrix"]
    assert matrix["petal_length"]["petal_width"] > 0.9


def test_meta_lists_mounted_groups_and_models(client):
    from app import __version__
    from app.ml import available_models

    meta = client.get("/api/v3/meta").json()
    assert meta["version"] == __version__
    assert meta["api_groups"] == ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]
    assert meta["available_models"] == available_models()


def test_dataset_catalog_registers_a_checksummed_version(client):
    record = client.post("/api/v3/datasets/register").json()
    assert record["rows"] == 150 and len(record["sha256"]) == 64
    assert "path" not in record  # o caminho do arquivo no servidor não sai pela API
    stored = DatasetCatalog(get_settings().data_dir).get("iris")
    assert Path(stored["path"]).is_relative_to(get_settings().data_dir)
    catalog = client.get("/api/v3/datasets").json()
    assert all("path" not in item for item in catalog)
    assert [item["id"] for item in catalog if item["name"] == "iris"] == [record["id"]]


def test_pipeline_dry_run_is_post_and_does_not_train(client):
    """Regressão: o console fazia GET numa rota POST (405)."""
    before = len(client.get("/api/v1/models").json())
    assert client.get("/api/v4/pipelines/dry-run").status_code == 405
    run = client.post("/api/v4/pipelines/dry-run", json={"model_name": "xgboost"}).json()
    assert [event["task"] for event in run["events"]] == [
        "ingest",
        "quality",
        "features",
        "anomalies",
        "train_plan",
    ]
    assert run["tasks"]["train_plan"]["rows"] == 150
    assert run["tasks"]["quality"]["passed"] is True
    assert len(client.get("/api/v1/models").json()) == before


def test_training_pipeline_persists_run_gate_and_report(client):
    result = client.post(
        "/api/v4/pipelines/train", json={"model_name": "gradient_boosting", "cv_folds": 3}
    ).json()
    assert result["status"] == "completed"
    assert result["release_gate"]["checks"][2]["observed"] == result["metrics"]["latency_ms"]
    report = Path(result["report_path"])
    assert report.is_file() and 'lang="pt-BR"' in report.read_text(encoding="utf-8")
    runs = client.get("/api/v4/pipelines/runs").json()
    assert runs[0]["id"] == result["pipeline_run_id"] and runs[0]["status"] == "completed"


def test_deployment_manifest_points_to_the_real_artifact(client, trained):
    """Regressão: o manifesto montava um nome de arquivo que não existia."""
    manifest = client.get("/api/v4/deployment/manifest", params={"model_name": "logistic_regression"}).json()
    record = client.get("/api/v1/models/logistic_regression").json()
    assert Path(manifest["spec"]["artifact"]).is_file()
    assert manifest["spec"]["checksum_sha256"] == record["checksum_sha256"]
    assert client.get("/api/v4/deployment/manifest", params={"model_name": "extra_trees"}).status_code == 404


def test_documentation_search_works(client):
    """Regressão: a rota usava `Path` sem importar e respondia 500."""
    response = client.post("/api/v5/studio/search", json={"query": "release gate promoção", "k": 3})
    assert response.status_code == 200
    results = response.json()
    assert 0 < len(results) <= 3
    assert all(item["id"] == "README.md" or item["id"].startswith("docs/") for item in results)


def test_frontier_and_bootstrap(client, trained):
    frontier = client.get("/api/v5/studio/frontier").json()
    assert frontier and all("latency_ms" in item["metrics"] for item in frontier)
    body = {"y_true": [0, 1, 1, 0, 1, 0], "y_pred": [0, 1, 0, 0, 1, 0], "iterations": 50}
    result = client.post("/api/v5/studio/evaluate-predictions", json=body).json()
    assert result["lower_95"] <= result["estimate"] <= result["upper_95"]
    assert result["estimate"] == pytest.approx(5 / 6)
    assert (
        client.post("/api/v5/studio/evaluate-predictions", json={"y_true": [0, 1], "y_pred": [0]}).status_code
        == 422
    )
    assert client.post("/api/v5/studio/evaluate-predictions", json={}).status_code == 422


def test_research_and_synthetic_data(client):
    matrix = client.post(
        "/api/v6/research/experiment-matrix", json={"models": ["naive_bayes"], "folds": 2}
    ).json()
    assert matrix["results"][0]["model"] == "naive_bayes"
    assert (
        client.post("/api/v6/research/experiment-matrix", json={"models": ["nao_existe"]}).status_code == 422
    )
    synthetic = client.post(
        "/api/v6/data/synthetic", json={"rows": 20, "features": 4, "informative": 2, "classes": 2}
    )
    assert synthetic.json()["rows"] == 20
    invalid = client.post("/api/v6/data/synthetic", json={"rows": 20, "features": 2, "informative": 5})
    assert invalid.status_code == 422, "entrada inválida é 422, não 500"
    space = client.get("/api/v6/research/model-space").json()
    assert "xgboost" in space["available"]


def test_robustness_evidence_pii_and_sbom(client, trained):
    stable = client.post(
        "/api/v6/modeling/robustness", params={"model_name": "logistic_regression", "scale": 0}
    ).json()
    assert stable["report"]["stability"] == 1.0
    assert client.post("/api/v6/modeling/robustness", params={"model_name": "extra_trees"}).status_code == 404
    version = client.get("/api/v1/models/logistic_regression").json()["version"]
    pack = client.post(
        "/api/v6/governance/evidence-pack", params={"model_name": "logistic_regression", "version": version}
    ).json()
    assert Path(pack["path"]).is_file() and len(pack["evidence_hash"]) == 64
    assert (
        client.post("/api/v6/governance/evidence-pack", params={"model_name": "x", "version": 1}).status_code
        == 404
    )
    scan = client.post("/api/v6/security/pii-scan", json={"text": "fale com pessoa@example.com"}).json()
    assert scan["report"]["contains_pii"] and "[REDACTED:email]" in scan["redacted"]
    assert any(
        c["name"].lower() == "fastapi" for c in client.get("/api/v6/platform/sbom").json()["components"]
    )


def test_forecast_accepts_the_minimum_series(client):
    """Regressão: o schema aceitava 8 pontos, mas o backtest exigia mais e a rota respondia 500."""
    for horizon in (1, 3, 30):
        response = client.post(
            "/api/v7/labs/forecast",
            json={"values": [1, 2, 3, 4, 5, 6, 7, 8], "horizon": horizon, "method": "naive"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["forecast"]["values"] == [8.0] * horizon
        assert body["backtest"]["windows"] >= 1
    assert client.post("/api/v7/labs/forecast", json={"values": [1.0] * 7}).status_code == 422
    assert (
        client.post("/api/v7/labs/forecast", json={"values": [1.0] * 9, "method": "magia"}).status_code == 422
    )


def test_text_and_automl_labs(client):
    payload = {
        "texts": ["great api", "great model", "bad api", "bad model"],
        "labels": ["good", "good", "bad", "bad"],
        "query": ["great api"],
    }
    assert client.post("/api/v7/labs/text", json=payload).json()["predictions"] == ["good"]
    mismatched = client.post("/api/v7/labs/text", json={**payload, "labels": ["good", "bad"]})
    assert mismatched.status_code == 422, "tamanhos diferentes eram 500"
    assert client.post("/api/v7/labs/text", json={**payload, "labels": ["good"] * 4}).status_code == 422
    automl = client.post("/api/v7/labs/automl", json={"max_models": 2, "folds": 2}).json()
    assert automl["task"] == "classification" and len(automl["results"]) == 2


def test_lineage_graph_comes_from_the_registry(client, trained):
    """Regressão: o grafo era fixo no código; agora reflete o registry."""
    graph = client.get("/api/v7/labs/lineage-graph").json()
    aliases = {f"model:{record['alias']}" for record in client.get("/api/v1/models").json()}
    assert aliases <= set(graph["nodes"])
    assert {
        "source": f"training_run:{trained['run_id']}",
        "relation": "produced",
        "target": "model:logistic_regression:v1",
    } in graph["edges"]
    assert any(
        edge["relation"] == "trained" and edge["source"].startswith("dataset:") for edge in graph["edges"]
    )
