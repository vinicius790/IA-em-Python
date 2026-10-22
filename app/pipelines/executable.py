from __future__ import annotations

from ..data import load_iris_dataframe
from ..features.engineering import enrich_iris
from ..monitoring.anomaly import detect_anomalies
from ..orchestration.dag import DAG, Task
from ..orchestration.runner import DAGRunner
from ..quality import validate_training_frame


def _quality(frame) -> dict:
    result = validate_training_frame(frame)
    return {"passed": result.passed, "score": result.score}


def _features(frame) -> dict:
    return {"columns": list(enrich_iris(frame).columns)}


def build_training_plan(model_name: str = "xgboost") -> DAG:
    """Plano de treino como DAG: valida e inspeciona os dados sem treinar nada."""
    dag = DAG(
        [
            Task("ingest", lambda c: load_iris_dataframe()),
            Task("quality", lambda c: _quality(c["ingest"]), depends_on=("ingest",)),
            Task("features", lambda c: _features(c["ingest"]), depends_on=("quality",)),
            Task("anomalies", lambda c: detect_anomalies(c["ingest"]), depends_on=("features",)),
            Task(
                "train_plan",
                lambda c: {
                    "model": model_name,
                    "rows": len(c["ingest"]),
                    "quality_passed": c["quality"]["passed"],
                    "anomalies": c["anomalies"]["count"],
                },
                depends_on=("anomalies",),
            ),
        ]
    )
    return dag


def dry_run(model_name: str = "xgboost") -> dict:
    run = DAGRunner().run(build_training_plan(model_name))
    run["tasks"].pop("ingest")  # o DataFrame bruto não vai na resposta
    return run
