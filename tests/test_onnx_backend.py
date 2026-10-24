"""Issue #10: exportação ONNX, backend por versão e paridade com scikit-learn."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from app.config import get_settings
from app.data import FEATURE_COLUMNS, load_iris_dataframe
from app.ml import build_models, load_artifact, predict_artifact, train_model
from app.registry import ModelRegistry
from app.services import MLService
from app.serving import onnx_backend
from app.serving.onnx_backend import (
    OnnxPredictor,
    UnsupportedModelError,
    benchmark_latency,
    export_onnx,
    onnx_available,
    predict_onnx,
    resolve_backend,
)

needs_onnx = pytest.mark.skipif(not onnx_available(), reason="extra `onnx` não instalado")

SUPPORTED_MODELS = [
    "logistic_regression",
    "naive_bayes",
    "svm",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "adaboost",
    "linear_sgd",
    "linear_discriminant",
    "quadratic_discriminant",
    "mlp",
    "stacking_ensemble",
]


@pytest.fixture(scope="module")
def iris():
    frame = load_iris_dataframe()
    return frame, frame[FEATURE_COLUMNS].astype(float).values.tolist()


def _registered(tmp_path, frame, model_name="logistic_regression", signing_key=""):
    result = train_model(frame, model_name, tmp_path, cv_folds=3)
    registry = ModelRegistry(tmp_path, signing_key)
    record = registry.register(model_name, result.artifact_path, {})
    return registry, record, load_artifact(result.artifact_path)


@needs_onnx
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_onnx_matches_sklearn(tmp_path, iris, model_name) -> None:
    frame, rows = iris
    artifact = load_artifact(train_model(frame, model_name, tmp_path, cv_folds=3).artifact_path)
    path = export_onnx(artifact, tmp_path / "model.onnx")
    expected_classes, expected_proba = predict_artifact(artifact, rows)
    classes, proba = predict_onnx(OnnxPredictor(path), artifact, rows)
    assert classes == expected_classes
    np.testing.assert_allclose(proba, expected_proba, atol=1e-5)


@pytest.mark.parametrize("model_name", ["knn", "voting_ensemble", "hist_gradient_boosting"])
def test_unsupported_model_raises_clear_error(tmp_path, model_name) -> None:
    artifact = {"model": build_models()[model_name], "features": FEATURE_COLUMNS}
    with pytest.raises(UnsupportedModelError, match="não suporta exportação ONNX"):
        export_onnx(artifact, tmp_path / "model.onnx")


def test_export_without_extra_raises(tmp_path, iris, monkeypatch) -> None:
    _, _, artifact = _registered(tmp_path, iris[0])
    monkeypatch.setattr(onnx_backend, "onnx_available", lambda: False)
    with pytest.raises(onnx_backend.OnnxUnavailableError, match=r"\[onnx\]"):
        export_onnx(artifact, tmp_path / "model.onnx")


def test_registry_onnx_checksum_and_signature(tmp_path, iris) -> None:
    registry, record, _ = _registered(tmp_path, iris[0], signing_key="k")
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"onnx-bytes")
    record = registry.attach_onnx(record["name"], record["version"], onnx_file)
    assert record["onnx_checksum_sha256"] and record["onnx_signature"]
    assert registry.verify_onnx(record)
    assert registry.verify(record)  # o pickle segue verificado independentemente
    onnx_file.write_bytes(b"tampered")
    assert not registry.verify_onnx(record)
    with pytest.raises(ValueError, match="inside the registry"):
        registry.attach_onnx(record["name"], record["version"], tmp_path.parent / "x.onnx")


def test_set_serving_backend_validation(tmp_path, iris) -> None:
    registry, record, _ = _registered(tmp_path, iris[0])
    with pytest.raises(ValueError, match="no ONNX artifact"):
        registry.set_serving_backend(record["name"], record["version"], "onnx")
    with pytest.raises(ValueError, match="Invalid backend"):
        registry.set_serving_backend(record["name"], record["version"], "tensorrt")
    assert resolve_backend(registry, record).backend == "sklearn"


def test_fallback_is_explicit(tmp_path, iris, caplog, monkeypatch) -> None:
    registry, record, _ = _registered(tmp_path, iris[0])
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"x")
    registry.attach_onnx(record["name"], record["version"], onnx_file)
    record = registry.set_serving_backend(record["name"], record["version"], "onnx")
    onnx_file.write_bytes(b"tampered")
    monkeypatch.setattr(onnx_backend, "onnx_available", lambda: True)
    with caplog.at_level(logging.WARNING, logger=onnx_backend.__name__):
        decision = resolve_backend(registry, record)
    assert decision.backend == "sklearn" and "checksum" in decision.fallback_reason
    assert "ONNX fallback" in caplog.text
    monkeypatch.setattr(onnx_backend, "onnx_available", lambda: False)
    assert "não instalado" in resolve_backend(registry, record).fallback_reason


@needs_onnx
def test_service_serves_onnx_version(tmp_path, iris) -> None:
    settings = get_settings().model_copy(update={"model_dir": tmp_path, "artifact_signing_key": ""})
    service = MLService(settings)
    _, record, artifact = _registered(tmp_path, iris[0])
    expected = service.score_dataframe(record["name"], iris[0])
    record = service.export_onnx(record["name"], record["version"], activate=True)
    assert record["serving_backend"] == "onnx"
    assert resolve_backend(service.registry, record).backend == "onnx"
    scored = service.score_dataframe(record["name"], iris[0])
    assert scored["prediction"].tolist() == expected["prediction"].tolist()
    np.testing.assert_allclose(scored["confidence"], expected["confidence"], atol=1e-5)
    report = benchmark_latency(
        artifact, OnnxPredictor(service.registry.onnx_file(record)), iris[1], repeats=5
    )
    assert set(report) == {"sklearn", "onnx"} and all("p95_ms" in v for v in report.values())
