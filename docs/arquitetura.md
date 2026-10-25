# Arquitetura

> 📘 **Diagramas Detalhados:** Para especificações completas de arquitetura de dados, ERD relacional, ciclo de vida dos modelos, formulação de drift detection e orquestração em DAG, consulte [ARCHITECTURE.md](ARCHITECTURE.md).

## Camadas

```text
Console web (app/console)    CLI (app/cli.py)    Clientes HTTP
              \                    |                  /
   FastAPI (app/main.py): middleware + routers de versão api_v1 … api_v8
                                   |
     Handlers por domínio (app/api/routers/: data, training, registry,
             inference, monitoring, governance, labs)
                                   |
   MLService (app/ml_service/ mixins; app/services.py é a fachada)
        /             |               |                 \
   ml.py         registry.py     monitoring/        governance/
 treino/tuning  versões, lock,   drift, SLO,        auditoria,
 métricas       checksum, HMAC   saúde              RBAC, evidências
        \             |               |                 /
   serving/ (ONNX, fallback sklearn)   modeling/ (Shapley)   tracing.py (OTel opcional)
                                   |
   SQLite via SQLAlchemy (schema só por Alembic)   arquivos em MODEL_DIR / ARTIFACT_DIR
```

- `app/main.py` monta a aplicação: request id, rate limit, métricas por template de rota, 500 sem
  detalhes e `BodySizeLimitMiddleware`. Todos os routers `/api` exigem `require_api_key` (papel por
  método e rota em `app/governance/rbac.py`); só `api_v1.public_router` (saúde e métricas) fica fora.
- `app/api/routers/` contém os handlers, agrupados por domínio; `api_v1.py` … `api_v8.py` montam as
  versões e mantêm helpers de compatibilidade. Rotas legadas com substituta v8 são marcadas `deprecated`.
- `app/ml_service/` divide as regras de negócio em mixins por domínio; `app/services.py` reexporta
  `MLService` e `get_service()`, a instância compartilhada por routers e CLI. É o único caminho que
  desserializa artefatos, sempre depois de `ModelRegistry.verify`.
- `app/registry.py` mantém `catalog.json` com versões e estágios sob lock de arquivo; o artefato ONNX
  tem SHA-256 e HMAC próprios. `app/serving/onnx_backend.py` serve ONNX com fallback para sklearn.
- `app/tracing.py` ativa OpenTelemetry só se as dependências estiverem instaladas.

## Fluxo de treino

1. O dataset é o Iris ou um CSV do catálogo (`dataset`); `validate_training_frame` aplica os gates de
   qualidade às features e ao alvo dele; falha interrompe o treino.
2. Divisão estratificada treino/teste, tuning opcional, validação cruzada no treino, ajuste final.
3. Métricas no teste (acurácia, F1, Brier, latência p95 de uma predição, importância por permutação).
4. Artefato, model card e lineage gravados; registro no registry como `candidate`.
5. Execução gravada em `training_runs` e `experiment_runs` (e no MLflow, se configurado); evento de auditoria.

## Fluxo de inferência

1. A versão servida é a de `production` ou, na falta dela, a mais recente não arquivada. Em
   `/api/v8/predict`, com um canário ativo, o hash da chave de roteamento escolhe entre a estável e a
   candidata (`app/deployment/canary.py`).
2. O artefato é verificado (checksum e, se configurada, assinatura) e mantido em cache enquanto o arquivo
   não mudar.
3. Itens do lote são agrupados por modelo e preditos de uma vez.
4. Cada predição vira uma linha em `prediction_logs`, base do monitoramento.

## Banco de dados

Tabelas: `iris_records`, `training_runs`, `prediction_logs` (com a versão que atendeu),
`experiment_runs`, `monitoring_reports`, `job_records`, `audit_events`, `pipeline_runs`, `api_users` e
`canary_deployments`. O schema é criado e evoluído só pelas migrações em
`app/migrations` (`init_db()` roda `alembic upgrade head` na inicialização). Um banco criado pelo antigo
`create_all()` da versão 7.0 é reconhecido e marcado na revisão `0003` antes de seguir.
