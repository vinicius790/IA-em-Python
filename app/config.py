from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import __version__

OPEN_ENVIRONMENT = "development"


class Settings(BaseSettings):
    """Configuração tipada, lida de variáveis de ambiente e do arquivo `.env`.

    Caminhos relativos são resolvidos a partir do diretório de trabalho do processo.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    app_name: str = "Iris AI Platform"
    environment: str = OPEN_ENVIRONMENT
    debug: bool = False
    database_url: str = "sqlite:///./data/iris.db"
    model_dir: Path = Path("models")
    artifact_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")
    api_key: str = ""
    artifact_signing_key: str = ""
    cors_origins: str = "http://localhost:8000"
    default_model: str = "xgboost"
    random_state: int = 42
    max_batch_size: int = Field(default=500, ge=1, le=10_000)
    max_request_bytes: int = Field(default=2_000_000, ge=1024, le=50_000_000)
    rate_limit_requests: int = Field(default=300, ge=1, le=100_000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)
    mlflow_tracking_uri: str = ""
    drift_psi_warning: float = Field(default=0.10, ge=0, le=10)
    drift_psi_threshold: float = Field(default=0.25, ge=0, le=10)
    gate_min_accuracy: float = Field(default=0.90, ge=0, le=1)
    gate_min_f1: float = Field(default=0.90, ge=0, le=1)
    gate_max_latency_ms: float = Field(default=100.0, gt=0)
    slo_p95_latency_ms: float = Field(default=100.0, gt=0)
    slo_min_mean_confidence: float = Field(default=0.70, ge=0, le=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.environment != OPEN_ENVIRONMENT and not self.api_key:
            raise ValueError(
                f"API_KEY is required when ENVIRONMENT={self.environment!r}; "
                f"only ENVIRONMENT={OPEN_ENVIRONMENT!r} may run without authentication."
            )
        if self.drift_psi_warning > self.drift_psi_threshold:
            raise ValueError("DRIFT_PSI_WARNING must not be greater than DRIFT_PSI_THRESHOLD.")
        return self

    @property
    def app_version(self) -> str:
        return __version__

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def ensure_directories(self) -> None:
        for directory in (self.model_dir, self.artifact_dir, self.data_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
