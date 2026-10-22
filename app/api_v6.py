"""API v6: pesquisa (matriz de experimentos), dados sintéticos, robustez, PII, SBOM e evidências."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .config import get_settings
from .data import load_iris_dataframe
from .data_catalog.synthetic import SyntheticSpec, generate
from .errors import invalid_input_as_422
from .governance.evidence import build_evidence_pack
from .governance.pii import redact_text, scan_text
from .ml import available_models
from .modeling.robustness import perturbation_report
from .platform.sbom import minimal_cyclonedx
from .research.experiment_matrix import ExperimentMatrix, ExperimentSpec
from .schemas import ExperimentRequest, PIIScanRequest, SyntheticRequest, check_cv_budget
from .services import get_service

router = APIRouter(prefix="/api/v6")
settings = get_settings()
service = get_service()


@router.post("/research/experiment-matrix", tags=["research"])
def experiment_matrix(payload: ExperimentRequest) -> dict:
    """Validação cruzada repetida, com o mesmo protocolo, para os modelos pedidos (ou todos)."""
    spec = ExperimentSpec(payload.models, payload.repeats, payload.folds, payload.random_state)
    with invalid_input_as_422():
        check_cv_budget(len(set(payload.models)) or len(available_models()), payload.folds, payload.repeats)
        return ExperimentMatrix(random_state=payload.random_state).run(load_iris_dataframe(), spec)


@router.get("/research/model-space", tags=["research"])
def model_space() -> dict:
    return {
        "available": available_models(),
        "registered": sorted({row["name"] for row in service.list_models()}),
    }


@router.post("/data/synthetic", tags=["data"])
def synthetic(payload: SyntheticRequest) -> dict:
    with invalid_input_as_422():
        frame, metadata = generate(SyntheticSpec(**payload.model_dump()))
    return {"metadata": metadata, "rows": len(frame), "preview": frame.head(8).to_dict(orient="records")}


@router.post("/modeling/robustness", tags=["modeling"])
def robustness(
    model_name: str = Query("xgboost", max_length=128), scale: float = Query(0.01, ge=0, le=2)
) -> dict:
    """Estabilidade das predições do modelo registrado sob ruído gaussiano nas features."""
    record = service.serving_record(model_name)
    artifact = service.load(record)
    with invalid_input_as_422():  # o dataset em que o modelo foi treinado (pode ter saído do catálogo)
        frame, columns, _ = service.reference_frame(model_name)
    features = frame[columns]
    return {"model": record["alias"], "report": perturbation_report(artifact["model"], features, scale=scale)}


@router.post("/security/pii-scan", tags=["security"])
def pii_scan(payload: PIIScanRequest) -> dict:
    return {"report": scan_text(payload.text), "redacted": redact_text(payload.text)}


@router.get("/platform/sbom", tags=["platform"])
def sbom() -> dict:
    return minimal_cyclonedx()


@router.post("/governance/evidence-pack", tags=["governance"])
def evidence_pack(model_name: str = Query(max_length=128), version: int = Query(ge=1)) -> dict:
    """Gera o pacote de evidências (registro + release gate + hash) de uma versão do registry."""
    record = service.registry.get(model_name, version)
    if record is None:
        raise HTTPException(404, "Model version not found in the registry")
    output = settings.artifact_dir / "governance" / f"evidence-{model_name}-v{version}.json"
    return build_evidence_pack(record, service.release_gate(record), output)
