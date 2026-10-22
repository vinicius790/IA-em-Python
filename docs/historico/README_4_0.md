# Iris AI Platform 4.0 — Aurora ML Fabric

> Documento histórico: README de uma versão anterior, mantido como registro. Várias capacidades descritas aqui foram removidas na 8.0 por serem esqueletos; veja o CHANGELOG.

Version 4.0 grows the original IA-em-Python into a full ML platform. Previous APIs remain intact; a new v3 control plane adds dataset governance, feature engineering, deterministic DAG orchestration, release gates, deployment manifests, AI copilot, retrieval baseline, operations console and infrastructure starters.

## v3 endpoints

- `GET /api/v3/meta`
- `GET /api/v3/data/profile`
- `GET /api/v3/data/features`
- `POST /api/v3/datasets/register`
- `POST /api/v3/pipelines/dry-run`
- `POST /api/v3/copilot/design`
- `POST /api/v3/copilot/explain-run`
- `GET /api/v3/health/score`
- `POST /api/v3/reports/demo`

## Tooling

`frontend/` is a zero-build console. `infra/` contains Kubernetes, Helm and Terraform starters. `configs/` and `pipelines/` provide declarative contracts for future external orchestrators.

## Principles

Preserve compatibility, record provenance, isolate optional dependencies, make releases auditable, fail safely, keep the local path simple and expose clear seams for distributed production infrastructure.
