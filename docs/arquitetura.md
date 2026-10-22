# Arquitetura

> 📘 **Diagramas Detalhados:** Para especificações completas de arquitetura de dados, ERD relacional, ciclo de vida dos modelos, formulação de drift detection e orquestração em DAG, consulte [ARCHITECTURE.md](ARCHITECTURE.md).

## Camadas

```text
Console web (app/console)    CLI (app/cli.py)    Clientes HTTP
              \                    |                  /
               FastAPI (app/main.py): middleware + routers api_v1 … api_v8
                                   |
                      MLService (app/services.py)
        /             |               |                 \
   ml.py         registry.py     monitoring/        governance/
 treino/tuning  versões, lock,   drift, SLO,        auditoria,
 métricas       checksum, HMAC   saúde              evidências
        \             |               |                 /
   SQLite via SQLAlchemy (schema só por Alembic)   arquivos em MODEL_DIR / ARTIFACT_DIR
```

- `app/main.py` monta a aplicação. O middleware HTTP define o request id, aplica o rate limit, registra
  métricas pelo template da rota e transforma exceções não tratadas em 500 sem detalhes. O
  `BodySizeLimitMiddleware` (ASGI) recusa corpos grandes. Todos os routers `/api` recebem a dependência
  `require_api_key`, que identifica a chave (mestra ou de usuário), confere o papel exigido pelo método e
  pela rota (`app/governance/rbac.py`) e deixa o usuário disponível para a auditoria; só
  `api_v1.public_router` (saúde e métricas) fica fora.
- `app/services.py` concentra as regras de negócio e é a única instância compartilhada (`get_service()`)
  usada pelos routers e pela CLI. É o único lugar que desserializa artefatos, sempre depois de
  `ModelRegistry.verify`.
- `app/ml.py` define o catálogo de modelos (`build_models`), o tuning e `train_model`, que grava o
  artefato `joblib` (modelo, label encoder e uma amostra de fundo para a explicação), o model card e o
  registro de lineage.
- `app/registry.py` mantém `catalog.json` em `MODEL_DIR` com versões e estágios, protegido por lock de
  arquivo; guarda só o nome do arquivo do artefato.

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
