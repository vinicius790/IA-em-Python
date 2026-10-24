"""Exportação ONNX e inferência com ONNX Runtime (issue #10).

Requer o extra opcional ``onnx`` (``skl2onnx`` + ``onnxruntime``). Sem ele, as
funções levantam :class:`OnnxUnavailableError`; quem decide o backend
(:func:`resolve_backend`) registra o fallback para scikit-learn de forma explícita.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

BACKENDS = ("sklearn", "onnx")

# Estimadores finais cuja conversão foi validada com paridade (atol=1e-5) no Iris.
SUPPORTED_ESTIMATORS = frozenset(
    {
        "AdaBoostClassifier",
        "CalibratedClassifierCV",
        "DecisionTreeClassifier",
        "ExtraTreesClassifier",
        "GaussianNB",
        "GradientBoostingClassifier",
        "LinearDiscriminantAnalysis",
        "LogisticRegression",
        "MLPClassifier",
        "QuadraticDiscriminantAnalysis",
        "RandomForestClassifier",
        "StackingClassifier",
    }
)
# Fora da lista: KNN com weights="distance" (diverge em float32 nos pontos de treino),
# HistGradientBoosting/Voting (sem conversor estável) e boosters externos.


class OnnxUnavailableError(RuntimeError):
    """O extra ``onnx`` não está instalado."""


class UnsupportedModelError(ValueError):
    """O modelo não tem conversão ONNX validada."""


def onnx_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("skl2onnx", "onnxruntime"))


def _require_onnx() -> None:
    if not onnx_available():
        raise OnnxUnavailableError("ONNX indisponível: instale o extra `pip install .[onnx]`.")


def final_estimator_name(model: Any) -> str:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    return type(estimator).__name__


def ensure_supported(model: Any) -> None:
    name = final_estimator_name(model)
    if name not in SUPPORTED_ESTIMATORS:
        supported = ", ".join(sorted(SUPPORTED_ESTIMATORS))
        raise UnsupportedModelError(f"Modelo '{name}' não suporta exportação ONNX. Suportados: {supported}.")


def export_onnx(artifact: dict[str, Any], path: str | Path) -> Path:
    """Converte o modelo do artefato para ONNX (entrada float32, saída de probabilidades)."""
    model = artifact["model"]
    ensure_supported(model)
    _require_onnx()
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    onnx_model = to_onnx(
        model,
        initial_types=[("input", FloatTensorType([None, len(artifact["features"])]))],
        options={id(estimator): {"zipmap": False}},
    )
    target = Path(path)
    target.write_bytes(onnx_model.SerializeToString())
    return target


class OnnxPredictor:
    """Sessão ONNX Runtime que devolve probabilidades como ``predict_proba``."""

    def __init__(self, path: str | Path) -> None:
        _require_onnx()
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self.session.get_inputs()[0].name
        outputs = [o.name for o in self.session.get_outputs()]
        self._proba = "probabilities" if "probabilities" in outputs else outputs[-1]

    def predict_proba(self, features: list[list[float]] | np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=np.float32)
        (proba,) = self.session.run([self._proba], {self._input: rows})
        return np.asarray(proba, dtype=np.float64)


def predict_onnx(
    predictor: OnnxPredictor, artifact: dict[str, Any], features: list[list[float]]
) -> tuple[list[str], np.ndarray]:
    """Mesmo contrato de ``app.ml.predict_artifact``, via ONNX Runtime."""
    probabilities = predictor.predict_proba(features)
    predictions = artifact["label_encoder"].inverse_transform(probabilities.argmax(axis=1))
    return [str(value) for value in predictions], probabilities


@dataclass(frozen=True)
class BackendDecision:
    backend: str
    onnx_path: Path | None = None
    fallback_reason: str | None = None


def resolve_backend(registry: Any, record: dict[str, Any]) -> BackendDecision:
    """Backend da versão (``serving_backend``); fallback para sklearn sempre logado."""
    requested = record.get("serving_backend", "sklearn")
    if requested != "onnx":
        return BackendDecision("sklearn")
    reason = None
    if not onnx_available():
        reason = "extra `onnx` não instalado"
    elif not registry.verify_onnx(record):
        reason = "model.onnx ausente ou com checksum/assinatura inválidos"
    if reason is None:
        return BackendDecision("onnx", onnx_path=registry.onnx_file(record))
    logger.warning("ONNX fallback para sklearn em %s: %s", record.get("alias"), reason)
    return BackendDecision("sklearn", fallback_reason=reason)


def benchmark_latency(
    artifact: dict[str, Any], predictor: OnnxPredictor, features: list[list[float]], repeats: int = 200
) -> dict[str, dict[str, float]]:
    """Latência por chamada (ms) de cada backend: p50 e p95."""
    import pandas as pd

    frame = pd.DataFrame(features, columns=artifact["features"])
    runners = {
        "sklearn": lambda: artifact["model"].predict_proba(frame),
        "onnx": lambda: predictor.predict_proba(features),
    }
    report: dict[str, dict[str, float]] = {}
    for backend, run in runners.items():
        run()
        samples = []
        for _ in range(repeats):
            start = perf_counter()
            run()
            samples.append((perf_counter() - start) * 1000)
        report[backend] = {
            "p50_ms": round(float(np.percentile(samples, 50)), 4),
            "p95_ms": round(float(np.percentile(samples, 95)), 4),
        }
    return report
