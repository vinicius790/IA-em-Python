# Iris AI Platform 7.0 — Apex AI Engineering Platform

> Documento histórico: README de uma versão anterior, mantido como registro. Várias capacidades descritas aqui foram removidas na 8.0 por serem esqueletos; veja o CHANGELOG.

Uma plataforma extensível de **Data Science, Machine Learning, MLOps, Model Serving, Analytics e IA aplicada**, construída a partir do repositório original `IA-em-Python` e preservando seus entry points legados.

A proposta da 7.0 é deixar de ser apenas um projeto de classificação Iris e tornar o repositório um **laboratório completo de engenharia de IA**, com componentes pequenos e combináveis em vez de um único monólito.

## Capacidades

| Domínio | Capacidades principais |
|---|---|
| Dados | catálogo local, contratos de dados, profiling, quality gates, ingestão CSV/JSON/JSONL/Parquet, dados sintéticos |
| Features | engenharia de atributos, manifestos, seleção estatística, feature store local |
| ML | 20 estratégias, ensembles, XGBoost, LightGBM/CatBoost opcionais, MLP, tuning Optuna, validação cruzada |
| Pesquisa | matriz de experimentos, repeated CV, métricas comparáveis, bootstrap CI, fronteira de Pareto |
| Registry | versões, stages, aliases, checksum SHA-256, lineage e model cards |
| Serving | inferência individual/batch, micro-batching, roteamento determinístico ponderado, benchmark |
| MLOps | pipelines DAG, retries, async jobs, auditoria, eventos, idempotência e workspaces |
| Monitoramento | drift PSI, anomalias, calibração, performance, alertas, health score, Prometheus |
| Governança | política de risco, release gates, canary/blue-green/rolling, rollback, RBAC, secrets |
| IA | providers LLM, copilot, RAG/hybrid search, prompt registry, guardrails e agente operacional limitado |
| Analytics | estatística descritiva, correlação, comparação de modelos e DuckDB opcional |
| Integrações | S3, Redis, Celery/Kombu, Ray/Dask, ONNX, SHAP/LIME, MLflow, Evidently, Prefect/Airflow |
| Plataforma | FastAPI/OpenAPI, CLI, console web, Docker Compose, Kubernetes, Helm, Terraform e CI/CD |
| Extensibilidade | plugins Python via entry points, adapters e contratos de integração |

## Arquitetura

```text
                              ┌───────────────────────────┐
                              │  Web Console / CLI / API  │
                              │      FastAPI v1...v6      │
                              └────────────┬──────────────┘
                                           │
                         ┌─────────────────┼─────────────────┐
                         ▼                 ▼                 ▼
                    Data Plane       ML Research       Control Plane
                 catalog/contracts   experiment matrix   registry/governance
                 quality/profiling   CV/CI/frontier      audit/events/jobs
                         │                 │                 │
                         └─────────────────┼─────────────────┘
                                           ▼
                                  Serving & Deployment
                            batch / microbatch / routing
                            canary / blue-green / rollback
                                           │
                   ┌───────────────────────┼───────────────────────┐
                   ▼                       ▼                       ▼
              Observability              AI Layer               Adapters
          metrics/drift/anomaly      RAG/copilot/agent      cloud/queue/ONNX
          health/calibration         prompts/guardrails      distributed/plugins
```

## APIs

A API mantém compatibilidade com as gerações anteriores e acrescenta superfícies novas:

```text
/v1   compatibilidade e endpoints originais
/v2   MLOps avançado
/v3   data/features/pipelines/copilot/health
/v4   control plane, governance, deployment, agent
/v5   Studio: search, frontier, bootstrap, workspace
/v6   Foundry: research, synthetic data, routing, microbatch, plugins, guardrails
/v7   Labs: AutoML, NLP, forecasting, schema inference e knowledge graph
```

Documentação OpenAPI automática em `/docs` e `/redoc`.

## Modelos disponíveis

O catálogo base inclui:

`logistic_regression`, `knn`, `naive_bayes`, `svm`, `calibrated_svm`, `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`, `hist_gradient_boosting`, `adaboost`, `linear_sgd`, `linear_discriminant`, `quadratic_discriminant`, `xgboost`, `mlp`, `voting_ensemble`, `stacking_ensemble`.

Com os extras analíticos, `lightgbm` e `catboost` também são adicionados.

## Uso rápido

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Abra `http://127.0.0.1:8000/console` para o console visual ou `http://127.0.0.1:8000/docs` para a API.

### Instalação do conjunto completo de extras

```bash
pip install -e '.[analytics,mlops,observability,api,cloud,distributed,ai,deep-learning,monitoring,workflow,rag,explainability,research,graphql,protocols]'
```

Integrações opcionais são descobertas em runtime e possuem fallback local quando a dependência externa não estiver instalada.

## CLI

```bash
iris-ai models
iris-ai stack
iris-ai quality
iris-ai train --model xgboost
iris-ai pipeline --model xgboost
iris-ai audit
iris-ai agent --goal "inspect model registry"
iris-ai export-openapi
```

## Scripts operacionais

```bash
python scripts/platform_audit.py
python scripts/health_report.py
python scripts/train_pipeline.py --model xgboost --tune
python scripts/batch_score.py data/processed/iris.csv --model xgboost
python scripts/export_openapi.py
```

Os scripts inserem automaticamente a raiz do repositório no `PYTHONPATH`, podendo ser executados diretamente a partir do clone.

## Qualidade e segurança

O repositório inclui testes automatizados, Ruff, Pylint, CodeQL, `pip-audit`, pre-commit, container não-root, healthcheck, limites de payload, API key opcional, rate limiting process-local, request IDs, proteção contra path traversal, verificação SHA-256 de modelos e políticas de release.

Ferramentas externas como Ruff, pip-audit, Kubernetes, Terraform, DVC e engines de workflow ficam configuradas para CI/produção e são opcionais no ambiente local.

## Estrutura

```text
app/
  automl/          AutoML tabular e inferência de tarefa
  nlp/             classificação de texto TF-IDF
  timeseries/      forecasting e backtesting baselines
  knowledge/       grafo leve de relações
  schema/           inferência semântica de schema
  adapters/       cloud, queue, distributed, ONNX, explainability
  agent/          ferramentas e agente operacional
  ai/             providers, RAG, prompting
  analytics/      estatística, comparação, warehouse
  data_catalog/   contratos, profiling, qualidade, ingestão, sintéticos
  deployment/     gates, manifests, rollout e rollback
  extensions/     catálogo de plugins
  features/       engineering, selection, store
  governance/     policy e audit
  llm/            guardrails
  modeling/       evaluation, frontier, SHAP
  monitoring/     drift, anomaly, calibration, performance, alerts
  orchestration/  DAG e runner
  pipelines/      blueprints e execução real
  platform/       eventos, idempotência, hashing, runtime, snapshot
  research/       experiment matrix
  routing/        model router
  serving/        batch, microbatch, benchmark
  workspace/      execução e manifestos
frontend/         console web
migrations/       Alembic
infra/            Kubernetes, Helm, Terraform
pipelines/        YAML + Airflow + Prefect
ops/              Prometheus + integrações opcionais
tests/            cobertura por subsistema
docs/             arquitetura, operações, governança, IA, pesquisa
```

## Princípio de extensibilidade

A plataforma separa **contratos** de **implementações**. Recursos pesados e serviços externos ficam atrás de adapters; o núcleo pode continuar executando em modo local. Isso permite começar com SQLite + arquivos e migrar gradualmente para PostgreSQL, Redis, S3, MLflow, object storage, filas, observabilidade distribuída e orquestradores externos.

## Licença

MIT.

## 7.0 — Apex Labs

A geração 7 adiciona uma camada de Data Science generalista sobre a infraestrutura MLOps: AutoML tabular, inferência de tarefa, classificação NLP TF-IDF, forecasting e backtesting de séries temporais, schema inference e um knowledge graph local para relações de datasets, pipelines e modelos.

Esses laboratórios não substituem frameworks especializados; eles formam baselines executáveis e pontos de integração que podem crescer para transformers, vision, speech, Spark, Ray, Feast, Kubeflow ou serviços externos por meio dos adapters existentes.
