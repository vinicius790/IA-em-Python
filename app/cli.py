from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from .config import get_settings
from .data import ensure_local_dataset, load_iris_dataframe
from .data_catalog.ingestion import ingest, write_dataset
from .data_catalog.profiling import profile_dataframe
from .database import SessionLocal, init_db
from .integrations import optional_stack_status
from .ml import TUNING_SPACES, available_models
from .services import get_service

app = typer.Typer(help="Iris AI Platform CLI", no_args_is_help=True)


def _dump(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


@app.command()
def models() -> None:
    """Lista os modelos disponíveis e quais aceitam tuning."""
    table = Table("Model", "Tunable")
    for name in available_models():
        table.add_row(name, "yes" if name in TUNING_SPACES else "no")
    print(table)


@app.command()
def stack() -> None:
    """Mostra quais integrações opcionais estão instaladas."""
    _dump(optional_stack_status())


@app.command()
def seed(replace: bool = typer.Option(False, help="Substitui as linhas existentes.")) -> None:
    """Carrega o dataset Iris na tabela `iris_records`."""
    init_db()
    with SessionLocal() as db:
        inserted = get_service().seed_database(db, replace=replace)
    _dump({"inserted": inserted})


@app.command()
def quality() -> None:
    """Roda os gates de qualidade do dataset."""
    _dump(get_service().quality_report())


@app.command()
def train(
    model: str = typer.Option("xgboost", help="Nome do modelo (veja `models`)."),
    test_size: float = typer.Option(0.2, min=0.06, max=0.49),
    cv_folds: int = typer.Option(5, min=2, max=20),
    tune: bool = typer.Option(False, "--tune", help="Busca de hiperparâmetros antes do treino."),
    trials: int = typer.Option(12, min=1, max=100, help="Tentativas da busca de hiperparâmetros."),
    dataset: str | None = typer.Option(None, help="Dataset do catálogo (`dataset-add`); padrão: Iris."),
) -> None:
    """Treina, avalia e registra um modelo."""
    init_db()
    with SessionLocal() as db:
        run = get_service().train(db, model, test_size, cv_folds, tune, trials, dataset=dataset)
    summary = {key: value for key, value in (run.metrics or {}).items() if isinstance(value, int | float)}
    _dump(
        {
            "run_id": run.id,
            "model_name": run.model_name,
            "status": run.status,
            "error": run.error_message,
            "metrics": summary,
            "optimization": (run.metrics or {}).get("optimization"),
            "artifact": run.artifact_path,
        }
    )
    if run.status != "completed":
        raise typer.Exit(1)


@app.command()
def pipeline(
    model: str = typer.Option("xgboost"),
    tune: bool = typer.Option(False, "--tune"),
    trials: int = typer.Option(12, min=1, max=100),
) -> None:
    """Pipeline completo: qualidade, treino, release gate e relatório HTML."""
    from .pipelines.service import TrainingPipeline

    init_db()
    with SessionLocal() as db:
        result = TrainingPipeline(get_service()).run(db, model, tune=tune, trials=trials)
    result.pop("metrics", None)
    _dump(result)
    if result["status"] != "completed":
        raise typer.Exit(1)


@app.command()
def dataset(path: Path = typer.Option(Path("data/processed/iris.csv"), help="Destino do CSV.")) -> None:
    """Materializa uma cópia CSV do Iris."""
    print(f"Dataset: {ensure_local_dataset(path)}")


@app.command()
def profile() -> None:
    """Perfil do dataset: tipos, nulos e estatísticas por coluna."""
    _dump(profile_dataframe(load_iris_dataframe()))


@app.command()
def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    model: str | None = typer.Option(None),
) -> None:
    """Prediz a classe usando a versão em produção (ou a mais recente) do modelo."""
    settings = get_settings()
    init_db()
    features = [sepal_length, sepal_width, petal_length, petal_width]
    with SessionLocal() as db:
        _dump(get_service().predict(db, model or settings.default_model, features))


@app.command()
def explain(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    model: str | None = typer.Option(None),
) -> None:
    """Explica a predição de uma instância com valores de Shapley exatos."""
    name = model or get_settings().default_model
    _dump(get_service().explain(name, [sepal_length, sepal_width, petal_length, petal_width]))


@app.command("batch-score")
def batch_score(
    input_path: Path,
    model: str | None = typer.Option(None),
    output: Path = typer.Option(Path("artifacts/predictions.csv")),
) -> None:
    """Pontua um arquivo CSV/JSON/JSONL/Parquet e grava as predições em CSV."""
    name = model or get_settings().default_model
    scored = get_service().score_dataframe(name, ingest(input_path))
    write_dataset(scored, output)
    print(f"{len(scored)} linhas gravadas em {output}")


@app.command()
def monitor(model: str | None = typer.Option(None)) -> None:
    """Resumo de monitoramento das predições recentes (confiança, latência, drift, saúde)."""
    init_db()
    with SessionLocal() as db:
        _dump(get_service().monitoring_summary(db, model or get_settings().default_model))


@app.command()
def audit(limit: int = typer.Option(50, min=1, max=500)) -> None:
    """Mostra os eventos de auditoria mais recentes."""
    from .governance.audit import recent_audit

    init_db()
    with SessionLocal() as db:
        _dump(recent_audit(db, limit))


@app.command("dataset-add")
def dataset_add(
    path: Path,
    name: str = typer.Option(..., help="Nome no catálogo: minúsculas, dígitos, `_` e `-`."),
    target: str | None = typer.Option(None, help="Coluna-alvo; sem ela, o alvo é inferido."),
) -> None:
    """Faz o perfil de um CSV, valida e registra o dataset no catálogo."""
    init_db()
    try:
        with SessionLocal() as db:
            record = get_service().register_dataset(db, name, path.read_text(encoding="utf-8-sig"), target)
    except (ValueError, OSError) as exc:
        typer.echo(f"Erro: {exc}")
        raise typer.Exit(1) from exc
    _dump({key: value for key, value in record.items() if key != "profile"})


@app.command()
def datasets() -> None:
    """Lista os datasets enviados ao catálogo."""
    table = Table("Dataset", "Linhas", "Alvo", "Features", "SHA-256")
    latest = {row["name"]: row for row in get_service().catalog().list() if row.get("target")}
    for row in latest.values():
        table.add_row(
            row["name"], str(row["rows"]), row["target"], str(len(row["features"])), row["sha256"][:12]
        )
    print(table)


@app.command()
def compare(
    models: list[str] = typer.Argument(..., help="Dois ou mais modelos (veja `models`)."),
    dataset: str | None = typer.Option(None, help="Dataset do catálogo; padrão: Iris."),
    folds: int = typer.Option(5, min=2, max=20),
    confidence: float = typer.Option(0.95, help="0.9, 0.95 ou 0.99."),
) -> None:
    """Compara modelos por validação cruzada estratificada e grava relatório JSON e HTML."""
    init_db()
    try:
        with SessionLocal() as db:
            report = get_service().compare(db, models, dataset=dataset, folds=folds, confidence=confidence)
    except ValueError as exc:
        typer.echo(f"Erro: {exc}")
        raise typer.Exit(1) from exc
    table = Table("#", "Modelo", "F1 macro", "IC", "Difere do melhor")
    for row in report["results"]:
        f1 = row["f1_macro"]
        table.add_row(
            str(row["rank"]),
            row["model"],
            f"{f1['mean']:.4f}",
            f"[{f1['lower']:.4f}; {f1['upper']:.4f}]",
            "sim" if row["differs_from_best"] else "não",
        )
    print(table)
    _dump(report["reports"])


@app.command("canary-start")
def canary_start(
    model: str,
    candidate_version: int,
    traffic: float = typer.Option(10.0, min=0.01, max=100, help="Percentual de tráfego da candidata."),
    min_requests: int = typer.Option(50, min=1, help="Predições da candidata antes da decisão."),
) -> None:
    """Abre um deploy canário entre a versão em produção e uma candidata."""
    from .deployment.canary import CanaryRule
    from .errors import CanaryConflictError

    service = get_service()
    init_db()
    try:
        with SessionLocal() as db:
            canary = service.start_canary(
                db, model, candidate_version, traffic, CanaryRule(min_requests=min_requests), created_by="cli"
            )
            _dump(service.canary_report(db, canary))
    except (KeyError, ValueError, CanaryConflictError) as exc:
        typer.echo(f"Erro: {exc}")
        raise typer.Exit(1) from exc


@app.command("canary-status")
def canary_status(
    canary_id: int,
    evaluate: bool = typer.Option(False, "--evaluate", help="Aplica a regra: promove ou reverte."),
) -> None:
    """Métricas de cada braço de um canário; com `--evaluate`, aplica a regra."""
    from .db_models import CanaryDeployment
    from .errors import CanaryConflictError

    service = get_service()
    init_db()
    with SessionLocal() as db:
        canary = db.get(CanaryDeployment, canary_id)
        if canary is None:
            typer.echo(f"Erro: canário {canary_id} não encontrado")
            raise typer.Exit(1)
        try:
            _dump(service.evaluate_canary(db, canary) if evaluate else service.canary_report(db, canary))
        except CanaryConflictError as exc:
            typer.echo(f"Erro: {exc}")
            raise typer.Exit(1) from exc


@app.command("user-create")
def user_create(
    name: str,
    role: str = typer.Option("reader", help="admin, operator ou reader."),
) -> None:
    """Cria um usuário da API e mostra a chave dele (uma única vez)."""
    from sqlalchemy.exc import IntegrityError

    from .db_models import ApiUser
    from .governance.audit import record_audit
    from .governance.rbac import ROLES, generate_key, hash_key
    from .schemas import UserCreate

    if role not in ROLES:
        typer.echo(f"Erro: papel inválido. Use um de: {', '.join(ROLES)}")
        raise typer.Exit(1)
    try:
        UserCreate(name=name, role=role)
    except ValueError as exc:
        typer.echo(f"Erro: nome de usuário inválido ({name!r})")
        raise typer.Exit(1) from exc
    init_db()
    key = generate_key()
    with SessionLocal() as db:
        db.add(ApiUser(name=name, role=role, key_hash=hash_key(key)))
        try:
            db.commit()
        except IntegrityError as exc:
            typer.echo(f"Erro: o usuário {name!r} já existe")
            raise typer.Exit(1) from exc
        record_audit(db, "user_created", actor="cli", user=name, role=role)
    _dump({"name": name, "role": role, "api_key": key})


@app.command()
def users() -> None:
    """Lista os usuários da API (sem as chaves, que não ficam guardadas)."""
    from sqlalchemy import select

    from .db_models import ApiUser

    init_db()
    table = Table("Id", "Nome", "Papel", "Ativo")
    with SessionLocal() as db:
        for user in db.scalars(select(ApiUser).order_by(ApiUser.id)).all():
            table.add_row(str(user.id), user.name, user.role, "sim" if user.active else "não")
    print(table)


@app.command("export-openapi")
def export_openapi(output: Path = typer.Option(Path("artifacts/openapi.json"))) -> None:
    """Exporta o contrato OpenAPI completo."""
    from .main import app as fastapi_app

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fastapi_app.openapi(), indent=2), encoding="utf-8")
    print(f"OpenAPI gravado em {output}")


@app.command("explain-pipeline")
def explain_pipeline_cmd(
    model: str = typer.Option("logistic_regression", help="Nome do modelo registrado."),
    repeats: int = typer.Option(5, min=2, max=20, help="Repetições do permutation importance."),
) -> None:
    """Executa o pipeline de explicabilidade (Feature Importance nativa + Permutação)."""
    from .data import FEATURE_COLUMNS, TARGET_COLUMN
    from .ml import get_model
    from .modeling.explainability import ExplainabilityPipeline

    df = load_iris_dataframe()
    X = df[FEATURE_COLUMNS]
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(df[TARGET_COLUMN])

    estimator = get_model(model)
    estimator.fit(X, y)

    pipeline = ExplainabilityPipeline(
        model=estimator,
        feature_names=FEATURE_COLUMNS,
        model_name=model,
        background_data=X,
    )
    report = pipeline.run_pipeline(X=X, y=y, n_repeats=repeats)

    table = Table("Ranking", "Feature", "Queda Acurácia (Média)", "Desvio Padrão")
    for idx, item in enumerate(report.permutation_importance, 1):
        table.add_row(
            f"#{idx}",
            item["feature"],
            f"{item['importance_mean']:.4f}",
            f"±{item['importance_std']:.4f}",
        )
    print(table)
    print(f"[bold green]{report.summary}[/bold green]")


@app.command("kfold")
def kfold_cmd(
    model: str = typer.Option("logistic_regression", help="Modelo para validação cruzada."),
    folds: int = typer.Option(5, min=2, max=20, help="Número de folds."),
) -> None:
    """Executa validação cruzada K-Fold com métricas detalhadas e intervalos de confiança."""
    from .data import FEATURE_COLUMNS, TARGET_COLUMN
    from .ml import get_model
    from .modeling.cross_validation import run_kfold_cross_validation

    df = load_iris_dataframe()
    X = df[FEATURE_COLUMNS]
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(df[TARGET_COLUMN])

    estimator = get_model(model)
    res = run_kfold_cross_validation(
        model=estimator,
        X=X,
        y=y,
        model_name=model,
        n_splits=folds,
        stratified=True,
    )

    table = Table("Métrica", "Média", "Desvio", "IC 95% (Inferior)", "IC 95% (Superior)")
    for name, m in res.aggregated_metrics.items():
        table.add_row(
            name,
            f"{m['mean']:.4f}",
            f"{m['std']:.4f}",
            f"{m['ci_lower']:.4f}",
            f"{m['ci_upper']:.4f}",
        )
    print(table)
    print(f"Validação {folds}-Fold concluída com sucesso para '{model}'. Amostras: {res.sample_count}")


@app.command("drift-check")
def drift_check_cmd(
    shift: float = typer.Option(0.5, help="Deslocamento artificial para testar a detecção de drift."),
    alpha: float = typer.Option(0.05, help="Nível de significância alfa para o KS-test."),
) -> None:
    """Executa verificação de Data Drift estatístico combinando PSI e KS-test."""
    from .data import FEATURE_COLUMNS
    from .monitoring.drift import statistical_drift_analysis

    df = load_iris_dataframe()
    reference = df[FEATURE_COLUMNS]
    current = reference.copy()
    current["petal_length"] = current["petal_length"] + shift

    analysis = statistical_drift_analysis(
        reference=reference,
        current=current,
        features=FEATURE_COLUMNS,
        ks_alpha=alpha,
    )

    table = Table("Feature", "PSI", "KS Estatística", "KS p-valor", "Status")
    for feat, data in analysis["features"].items():
        table.add_row(
            feat,
            f"{data['psi']:.4f}",
            f"{data['ks_statistic']:.4f}",
            f"{data['ks_p_value']:.4f}",
            data["status"],
        )
    print(table)
    status_color = "red" if analysis["status"] == "drift" else ("yellow" if analysis["status"] == "warning" else "green")
    print(f"[{status_color}]Status Geral de Drift: {analysis['status'].upper()} (Max PSI: {analysis['max_psi']:.4f})[/{status_color}]")


@app.command("audit-export")
def audit_export_cmd(
    model: str = typer.Option("logistic_regression", help="Nome do modelo."),
    output: Path = typer.Option(Path("artifacts/model_audit_report.md"), help="Destino do arquivo."),
    format: str = typer.Option("markdown", help="Formato de exportação: markdown ou json."),
) -> None:
    """Gera e exporta relatório completo de auditoria e governança do modelo."""
    from .data import FEATURE_COLUMNS, TARGET_COLUMN
    from .governance.model_audit import build_model_audit_report, export_audit_report
    from .ml import get_model

    df = load_iris_dataframe()
    X = df[FEATURE_COLUMNS]
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(df[TARGET_COLUMN])

    estimator = get_model(model)
    estimator.fit(X, y)

    report = build_model_audit_report(
        model_name=model,
        model=estimator,
        X=X,
        y=y,
        n_splits=5,
    )

    exported_path = export_audit_report(report, output, format=format)
    print(f"[bold green]Relatório de auditoria ({format}) exportado com sucesso para: {exported_path}[/bold green]")
    print(f"Veredito de Governança: [bold]{report.overall_status}[/bold]")


if __name__ == "__main__":
    app()

