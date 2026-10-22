"""Regras de negócio dos módulos de dados, monitoramento, orquestração e labs."""

import numpy as np
import pandas as pd
import pytest

from app.automl.engine import AutoMLConfig, TabularAutoML
from app.automl.task import infer_task
from app.data import FEATURE_COLUMNS, load_iris_dataframe
from app.data_catalog.catalog import DatasetCatalog
from app.data_catalog.contracts import ColumnContract, DatasetContract
from app.data_catalog.profiling import profile_dataframe
from app.data_catalog.synthetic import SyntheticSpec, generate
from app.features.engineering import ENGINEERED_COLUMNS, enrich_iris
from app.governance.pii import redact_text, scan_text
from app.knowledge.graph import KnowledgeGraph
from app.model_selection import leaderboard, pareto_frontier
from app.modeling.evaluation import bootstrap_metric
from app.monitoring.calibration import multiclass_brier
from app.monitoring.drift import drift_report, psi
from app.monitoring.health import score_health
from app.monitoring.sla import SLO, SLOEvaluator
from app.orchestration.dag import DAG, DAGError, Task
from app.orchestration.runner import DAGRunner
from app.quality import validate_training_frame
from app.research.experiment_matrix import ExperimentMatrix, ExperimentSpec
from app.search.hybrid import HybridSearch, SearchDocument
from app.search.ingest import documentation_files
from app.timeseries.baselines import backtest, forecast


def test_quality_gates_pass_for_iris_and_name_each_failure():
    frame = load_iris_dataframe()
    assert validate_training_frame(frame).passed

    broken = frame.copy()
    broken.loc[0, "sepal_length"] = np.nan
    broken.loc[1, "petal_width"] = -1.0
    failed = {check["name"] for check in validate_training_frame(broken).checks if not check["passed"]}
    assert failed == {"missing_values", "positive_measurements"}

    result = validate_training_frame(frame.drop(columns=["petal_width"]).head(10))
    assert {c["name"] for c in result.checks if not c["passed"]} == {"required_columns", "min_rows"}
    assert validate_training_frame(pd.concat([frame, frame])).passed is False  # linhas duplicadas


def test_profile_reports_types_nulls_and_ranges():
    profile = profile_dataframe(pd.DataFrame({"x": [1.0, 2.0, None], "label": ["a", "b", "a"]}))
    x, label = profile["column_profiles"]
    assert (x["semantic_type"], x["nulls"], x["numeric"]["max"]) == ("numeric", 1, 2.0)
    assert (label["semantic_type"], label["top_values"]) == ("categorical", {"a": 2, "b": 1})
    assert profile["rows"] == 3


def test_psi_is_zero_for_identical_data_and_grows_with_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 2000)
    assert psi(reference, reference) == pytest.approx(0.0, abs=1e-9)
    assert psi(reference, rng.normal(0, 1, 2000)) < 0.05
    assert psi(reference, rng.normal(3, 1, 2000)) > 1.0
    assert psi(reference, np.full(100, 50.0)) > 1.0, "valores fora da faixa da referência contam como drift"
    assert psi(np.array([]), reference) == 0.0


def test_drift_report_classifies_each_feature():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(0, 1, 1000)})
    current = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(2, 1, 1000)})
    report = drift_report(reference, current, ["a", "b"])
    assert report["features"]["a"]["status"] == "stable"
    assert report["drifted_features"] == ["b"]
    assert report["status"] == "drift"
    assert report["max_psi"] == report["features"]["b"]["psi"]


def test_health_score_explains_every_penalty():
    assert score_health(quality_passed=True, drift_status="stable", failed_slos=[])["score"] == 100.0
    assert (
        score_health(quality_passed=True, drift_status="insufficient_data", failed_slos=[])["status"]
        == "healthy"
    )
    degraded = score_health(quality_passed=True, drift_status="drift", failed_slos=["p95_latency_ms"])
    assert (degraded["score"], degraded["status"]) == (50.0, "degraded")
    assert [p["reason"] for p in degraded["penalties"]] == ["drift_drift", "slo_failed:p95_latency_ms"]
    assert score_health(quality_passed=False, drift_status="drift", failed_slos=["a"])["status"] == "critical"


def test_slo_evaluator_handles_direction_and_missing_metrics():
    slos = [SLO("availability", 0.99), SLO("latency", 100, "lte"), SLO("accuracy", 0.9)]
    result = SLOEvaluator(slos).evaluate({"availability": 0.995, "latency": 120})
    by_name = {check["name"]: check for check in result["checks"]}
    assert by_name["availability"]["passed"] is True
    assert by_name["latency"]["passed"] is False
    assert by_name["accuracy"]["status"] == "unknown"
    assert result["passed"] is False


def test_brier_score_bounds():
    classes = ["a", "b"]
    assert multiclass_brier(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]), classes) == 0.0
    assert multiclass_brier(["a", "b"], np.array([[0.0, 1.0], [1.0, 0.0]]), classes) == 2.0
    assert multiclass_brier(["a"], np.array([[0.5, 0.5]]), classes) == pytest.approx(0.5)


def test_engineered_features_accept_frames_and_arrays():
    frame = load_iris_dataframe()
    enriched = enrich_iris(frame)
    assert list(enriched.columns[-5:]) == ENGINEERED_COLUMNS
    assert enriched.loc[0, "petal_area"] == pytest.approx(1.4 * 0.2)
    assert len(frame.columns) == 5, "o DataFrame original não é alterado"
    from_array = enrich_iris(frame[FEATURE_COLUMNS].to_numpy())
    assert from_array["sepal_area"].tolist() == enriched["sepal_area"].tolist()


def test_dag_orders_tasks_retries_and_rejects_cycles():
    attempts = []

    def flaky(_):
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("falha transitória")
        return 1

    dag = DAG([Task("b", lambda c: c["a"] + 1, depends_on=("a",)), Task("a", flaky, retries=1)])
    run = DAGRunner().run(dag)
    assert run["tasks"] == {"a": 1, "b": 2}
    assert [event["status"] for event in run["events"]] == ["retry", "completed", "completed"]
    cyclic = DAG([Task("a", lambda c: 1, depends_on=("b",)), Task("b", lambda c: 1, depends_on=("a",))])
    with pytest.raises(DAGError, match="Cycle"):
        cyclic.validate()
    with pytest.raises(DAGError, match="Missing dependency"):
        DAG([Task("a", lambda c: 1, depends_on=("x",))]).validate()


def test_dataset_contract_and_catalog(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(path, index=False)
    catalog = DatasetCatalog(tmp_path)
    record = catalog.register("x", path, DatasetContract("x", [ColumnContract("x", "int64", False)]))
    assert record["rows"] == 3 and len(record["sha256"]) == 64
    assert catalog.register("x", path)["id"] == record["id"], "mesmo conteúdo, mesma versão"
    assert len(catalog.list("x")) == 1
    strict = DatasetContract("x", [ColumnContract("x", "int64", False, max_value=2)])
    with pytest.raises(ValueError, match="contract failed"):
        catalog.register("x", path, strict)


def test_forecast_methods_and_backtest_window():
    values = [float(v) for v in range(20)]
    assert forecast(values, 3, "naive").values == [19.0, 19.0, 19.0]
    assert forecast(values, 2, "linear_trend").values == pytest.approx([20.0, 21.0])
    assert forecast(values, 1, "moving_average", window=4).values == [17.5]
    with pytest.raises(ValueError, match="Unknown method"):
        forecast(values, 1, "magia")
    # Regressão: o backtest ignorava `window`; janelas diferentes têm de dar erros diferentes.
    short = backtest(values, 1, "moving_average", min_train=8, window=1)
    long = backtest(values, 1, "moving_average", min_train=8, window=8)
    assert short["mae"] == pytest.approx(1.0)
    assert long["mae"] == pytest.approx(4.5)
    with pytest.raises(ValueError, match="not enough"):
        backtest(values[:9], 1, "naive", min_train=8)


def test_automl_task_inference_and_regression():
    assert infer_task(load_iris_dataframe(), "class_name").task == "classification"
    frame = pd.DataFrame({"x": range(12), "z": [1, 2] * 6})
    frame["target"] = frame["x"] * 2.5 + frame["z"]
    result = TabularAutoML(AutoMLConfig(target="target", max_models=2, folds=2)).run(frame)
    assert result["task"] == "regression"
    assert result["results"][0]["r2"] >= result["results"][1]["r2"]


def test_experiment_matrix_uses_the_same_protocol_for_all_models():
    result = ExperimentMatrix().run(
        load_iris_dataframe(), ExperimentSpec(["naive_bayes", "knn"], folds=2, repeats=1)
    )
    assert [row["model"] for row in result["results"]] == ["naive_bayes", "knn"]
    assert all(row["folds"] == 2 and 0.8 <= row["accuracy_mean"] <= 1 for row in result["results"])
    with pytest.raises(ValueError, match="Unknown models"):
        ExperimentMatrix().run(load_iris_dataframe(), ExperimentSpec(["nao_existe"]))


def test_synthetic_generation_validates_spec():
    frame, meta = generate(SyntheticSpec(rows=25, features=4, informative=2, classes=2))
    assert frame.shape == (25, 5) and meta["target"] == "target"
    with pytest.raises(ValueError, match="informative"):
        generate(SyntheticSpec(features=2, informative=5))


def test_leaderboard_and_pareto_frontier():
    records = [
        {"name": "rapido", "version": 1, "metrics": {"f1_macro": 0.90, "latency_ms": 1.0}},
        {"name": "preciso", "version": 1, "metrics": {"f1_macro": 0.99, "latency_ms": 9.0}},
        {"name": "dominado", "version": 1, "metrics": {"f1_macro": 0.89, "latency_ms": 5.0}},
        {"name": "sem_medida", "version": 1, "metrics": {"f1_macro": 1.0}},
    ]
    assert [row["name"] for row in leaderboard(records, "f1_macro")][:2] == ["sem_medida", "preciso"]
    assert leaderboard(records, "inexistente")[0]["value"] is None
    assert {record["name"] for record in pareto_frontier(records)} == {"rapido", "preciso"}


def test_bootstrap_interval_contains_the_estimate():
    result = bootstrap_metric([0, 1, 1, 0] * 10, [0, 1, 0, 0] * 10, iterations=100)
    assert result["estimate"] == 0.75
    assert result["lower_95"] <= 0.75 <= result["upper_95"]


def test_pii_scan_and_redaction():
    source = "Contato: pessoa@example.com, CPF 123.456.789-09"
    report = scan_text(source)
    assert {finding["category"] for finding in report["findings"]} >= {"email", "cpf_like"}
    redacted = redact_text(source)
    assert "pessoa@example.com" not in redacted and "123.456.789-09" not in redacted
    assert scan_text("texto sem dados pessoais")["contains_pii"] is False


def test_pii_scan_is_linear_on_adversarial_text():
    # Regressão: o regex de e-mail levava minutos (custo quadrático) com 100 kB assim.
    from time import perf_counter

    started = perf_counter()
    for text in ("a." * 50_000, "1-" * 50_000):
        scan_text(text)
        redact_text(text)
    assert perf_counter() - started < 5
    assert redact_text("x joao.silva+tag@mail.example.com y") == "x [REDACTED:email] y"


def test_graph_shortest_path():
    graph = KnowledgeGraph()
    graph.add_edge("a", "r", "b")
    graph.add_edge("b", "r", "c")
    assert graph.shortest_path("a", "c") == ["a", "b", "c"]
    assert graph.shortest_path("c", "a") == []


def test_search_ranks_the_matching_document_first(tmp_path):
    index = HybridSearch(
        [
            SearchDocument("1", "model registry release gates", {}),
            SearchDocument("2", "database migrations", {}),
        ]
    )
    assert index.search("registry")[0].id == "1"
    assert HybridSearch().search("qualquer") == []
    # A indexação só enxerga README.md e docs/, nunca o repositório inteiro.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guia.md").write_text("guia", encoding="utf-8")
    (tmp_path / "README.md").write_text("leia-me", encoding="utf-8")
    (tmp_path / ".env").write_text("API_KEY=segredo", encoding="utf-8")
    (tmp_path / "outro").mkdir()
    (tmp_path / "outro" / "nota.md").write_text("fora", encoding="utf-8")
    assert [p.name for p in documentation_files(tmp_path)] == ["README.md", "guia.md"]
