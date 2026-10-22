# Iris AI Platform 5.0 — Aurora Control Plane

> Documento histórico: README de uma versão anterior, mantido como registro. Várias capacidades descritas aqui foram removidas na 8.0 por serem esqueletos; veja o CHANGELOG.

This release is intentionally much broader than a model-demo repository. It keeps the original Iris API and the 3.x MLOps layer while adding a control plane that can evolve into a production ML platform.

## Major subsystems

### Data
Dataset contracts, quality gates, version catalog, profiles, CSV/JSON/JSONL/Parquet ingestion and feature manifests.

### Feature engineering
Derived Iris features, generic preprocessing, variance filtering and ANOVA/mutual-information selection.

### ML operations
The previous model zoo, training, tuning, registry, checksums, lineage, model cards and experiment tracking remain intact.

### Monitoring
PSI drift, prediction performance, health scoring, Isolation Forest anomaly detection, calibration/reliability analysis, group-performance primitives and alert rules.

### Governance
Release policies, risk classification and append-only audit events.

### Orchestration
Reusable DAGs, executable local training plans, retries, event history and pipeline blueprints for classification, batch scoring and retraining.

### Deployment
Canary/blue-green/rolling rollout plans with explicit rollback paths.

### AI
Provider abstraction, offline copilot, bounded operations agent and deterministic retrieval baseline. The agent is restricted to registered tools and supports dry-run execution.

### Infrastructure
Kubernetes, Helm, Terraform and a full local Compose topology with PostgreSQL, Redis, MinIO, Prometheus and Grafana.

## API v4

`/api/v4/system/capabilities`, `/api/v4/data/quality`, `/api/v4/data/anomalies`, `/api/v4/features/selection`, `/api/v4/pipelines/blueprint/{name}`, `/api/v4/pipelines/dry-run`, `/api/v4/deploy/plan`, `/api/v4/governance/evaluate`, `/api/v4/audit`, `/api/v4/agent/tools`, `/api/v4/agent/execute`, `/api/v4/features/manifest` and `/api/v4/runtime/profile`.

## Architecture rule

5.0 is intentionally additive. Expensive distributed services are integration points, not mandatory local prerequisites. This makes the repository useful as a research sandbox, portfolio project and starting point for a real platform.
