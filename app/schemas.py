from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .deployment.canary import ARM_WINDOW
from .modeling.intervals import CONFIDENCE_LEVELS

Stage = Literal["candidate", "staging", "production", "archived"]
# Teto de treinos de validação cruzada (modelos x dobras x repetições) numa requisição síncrona:
# as rotas de comparação, matriz de experimentos e AutoML seguram um worker enquanto treinam.
MAX_CV_FITS = 60


def check_cv_budget(models: int, folds: int, repeats: int = 1) -> None:
    fits = models * folds * repeats
    if fits > MAX_CV_FITS:
        raise ValueError(
            f"Request needs {fits} cross-validation fits (models x folds x repeats); the limit is {MAX_CV_FITS}"
        )


class IrisFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sepal_length: float = Field(gt=0, le=20)
    sepal_width: float = Field(gt=0, le=20)
    petal_length: float = Field(gt=0, le=20)
    petal_width: float = Field(gt=0, le=20)

    def as_list(self) -> list[float]:
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class IrisCreate(IrisFeatures):
    class_name: str = Field(min_length=1, max_length=64)


class IrisRead(IrisCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    source: str = "sklearn"
    created_at: datetime


class PredictionRequest(IrisFeatures):
    model_name: str | None = Field(default=None, max_length=128)


class PredictionResponse(BaseModel):
    prediction_id: int
    model_name: str
    model_version: int
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    request_id: str | None = None
    latency_ms: float | None = None


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(min_length=1, max_length=10_000)


class FeedbackRequest(BaseModel):
    actual_class: str = Field(min_length=1, max_length=64)


class TrainingRequest(BaseModel):
    model_name: str = Field(default="xgboost", max_length=128)
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    tune_hyperparameters: bool = False
    cv_folds: int = Field(default=5, ge=2, le=20)
    optimization_trials: int = Field(default=12, ge=1, le=100)
    # Nome de um dataset enviado (catálogo); vazio ou "iris" treina no Iris.
    dataset: str | None = Field(default=None, max_length=64)


class TrainingResponse(BaseModel):
    run_id: int
    model_name: str
    status: str
    metrics: dict[str, Any] | None = None
    artifact_path: str | None = None
    dataset_hash: str | None = None
    lineage_hash: str | None = None
    model_card_path: str | None = None
    error_message: str | None = None


class ModelInfo(BaseModel):
    name: str
    version: int
    alias: str
    stage: str
    artifact_path: str
    checksum_sha256: str
    metrics: dict[str, Any]
    created_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    database: bool
    model_available: bool


class DatasetSummary(BaseModel):
    rows: int
    features: list[str]
    classes: dict[str, int]


class TrainingRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    model_name: str
    status: str
    metrics: dict[str, Any] | None
    parameters: dict[str, Any] | None
    artifact_path: str | None
    error_message: str | None
    duration_seconds: float | None
    dataset_hash: str | None = None
    lineage_hash: str | None = None
    model_card_path: str | None = None
    created_at: datetime


class DataQualityResponse(BaseModel):
    passed: bool
    score: float
    checks: list[dict[str, Any]]
    profile: dict[str, Any]


class DriftResponse(BaseModel):
    model_name: str
    status: str
    report: dict[str, Any]


class ExplainResponse(BaseModel):
    model_name: str
    model_version: int
    explanation: dict[str, Any]


class PromotionRequest(BaseModel):
    version: int = Field(ge=1)
    stage: Stage
    notes: str = Field(default="", max_length=1000)
    force: bool = False


class ExperimentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    backend: str
    params: dict[str, Any] | None
    metrics: dict[str, Any] | None
    tags: dict[str, str] | None
    external_run_id: str | None
    created_at: datetime


class AsyncTrainingResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    created_at: datetime


class JobRead(BaseModel):
    id: str
    kind: str
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: dict[str, Any] | None
    error: str | None


class PlatformStatus(BaseModel):
    version: str
    environment: str
    auth_required: bool
    model_count: int
    database: bool
    tracking_backend: str
    optional_stack: dict[str, Any]


class MonitoringReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    model_name: str | None
    report_type: str
    status: str
    payload: dict[str, Any]
    created_at: datetime


class PipelineRequest(BaseModel):
    model_name: str = Field(default="xgboost", max_length=128)
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    cv_folds: int = Field(default=5, ge=2, le=20)
    tune: bool = False
    trials: int = Field(default=12, ge=1, le=100)


class DryRunRequest(BaseModel):
    model_name: str = Field(default="xgboost", max_length=128)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


class EvaluatePredictionsRequest(BaseModel):
    y_true: list[str | int] = Field(min_length=2, max_length=100_000)
    y_pred: list[str | int] = Field(min_length=2, max_length=100_000)
    metric: Literal["accuracy", "f1_macro"] = "accuracy"
    iterations: int = Field(default=300, ge=10, le=5000)
    seed: int = 42

    @model_validator(mode="after")
    def _same_length(self) -> EvaluatePredictionsRequest:
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        return self


class ExperimentRequest(BaseModel):
    # Vazio = todos os modelos do catálogo; o teto de treinos é conferido na rota, que sabe quantos são.
    models: list[str] = Field(default_factory=list, max_length=30)
    folds: int = Field(default=3, ge=2, le=10)
    repeats: int = Field(default=1, ge=1, le=5)
    random_state: int = 42


class SyntheticRequest(BaseModel):
    task: Literal["classification", "regression"] = "classification"
    rows: int = Field(default=200, ge=10, le=100_000)
    features: int = Field(default=8, ge=2, le=500)
    informative: int = Field(default=5, ge=1, le=500)
    classes: int = Field(default=3, ge=2, le=20)
    noise: float = Field(default=0.15, ge=0)
    random_state: int = 42


class PIIScanRequest(BaseModel):
    text: str = Field(max_length=100_000)


class AutoMLRequest(BaseModel):
    max_models: int = Field(default=6, ge=1, le=20)
    folds: int = Field(default=3, ge=2, le=10)

    @model_validator(mode="after")
    def _within_budget(self) -> AutoMLRequest:
        check_cv_budget(self.max_models, self.folds)
        return self


class TextRequest(BaseModel):
    texts: list[str] = Field(min_length=2, max_length=1000)
    labels: list[str] = Field(min_length=2, max_length=1000)
    query: list[str] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def _aligned(self) -> TextRequest:
        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have equal length")
        if len(set(self.labels)) < 2:
            raise ValueError("labels must contain at least two distinct classes")
        return self


class ForecastRequest(BaseModel):
    values: list[float] = Field(min_length=8, max_length=10_000)
    horizon: int = Field(default=3, ge=1, le=365)
    method: Literal["naive", "moving_average", "linear_trend"] = "linear_trend"
    window: int = Field(default=5, ge=1, le=1000)


class NamedPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = Field(min_length=1, max_length=128)
    # Finitos: NaN e infinito chegariam ao modelo e virariam 500 (ou NaN gravado no log).
    features: dict[str, FiniteFloat] = Field(min_length=1, max_length=500)
    # Chave que fixa o braço do canário para um mesmo cliente; sem ela vale o id da requisição.
    routing_key: str | None = Field(default=None, min_length=1, max_length=128)


class NamedExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = Field(min_length=1, max_length=128)
    features: dict[str, FiniteFloat] = Field(min_length=1, max_length=500)


class ComparisonRequest(BaseModel):
    models: list[str] = Field(min_length=2, max_length=30)
    dataset: str | None = Field(default=None, max_length=64)
    folds: int = Field(default=5, ge=2, le=20)
    confidence: float = 0.95
    random_state: int | None = None

    @model_validator(mode="after")
    def _known_confidence(self) -> ComparisonRequest:
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"confidence must be one of: {', '.join(map(str, CONFIDENCE_LEVELS))}")
        check_cv_budget(len(set(self.models)), self.folds)
        return self


class CanaryRuleRequest(BaseModel):
    min_requests: int = Field(default=50, ge=1, le=ARM_WINDOW)
    max_confidence_drop: float = Field(default=0.05, ge=0, le=1)
    max_latency_ratio: float = Field(default=1.5, ge=1, le=100)
    max_accuracy_drop: float = Field(default=0.02, ge=0, le=1)
    min_labeled: int = Field(default=20, ge=1, le=ARM_WINDOW)


class CanaryRequest(BaseModel):
    model_name: str = Field(min_length=1, max_length=128)
    candidate_version: int = Field(ge=1)
    traffic_percent: float = Field(default=10, gt=0, le=100)
    rule: CanaryRuleRequest = Field(default_factory=CanaryRuleRequest)


class CanaryTrafficRequest(BaseModel):
    traffic_percent: float = Field(gt=0, le=100)


class UserCreate(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    role: Literal["admin", "operator", "reader"]

    @model_validator(mode="after")
    def _not_reserved(self) -> UserCreate:
        if self.name in {"api-key", "anonymous", "api", "system"}:
            raise ValueError(f"'{self.name}' is a reserved name")
        return self


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    role: str
    active: bool
    created_at: datetime
