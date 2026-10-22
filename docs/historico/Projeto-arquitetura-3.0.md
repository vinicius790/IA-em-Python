# Arquitetura da versão 3.0 (histórico)

> Documento histórico: descreve o projeto na versão 3.0 e não corresponde ao código atual.

```text
Arquitetura da plataforma · Iris AI Platform 3.0

app/
├── API: FastAPI v1 + v2 / OpenAPI
├── Data: Pandas + adaptadores Polars/DuckDB
├── Quality: profiling, gates, PSI drift
├── ML: 20 estratégias, CV, métricas, ensembles
├── Optimization: Optuna/TPE + tuning sklearn
├── Registry: versões, aliases, stages, SHA-256
├── MLOps: MLflow opcional, lineage, model cards
├── Explainability: permutation importance
├── Jobs: treinamento assíncrono
├── Observability: Prometheus, Grafana, structlog, OTel opcional
├── Security: API key, rate limit, request size, request ID, path guard
└── Compatibility: Main.py/main.py/neural_network.py/predict.py

data/
├── contract.json
├── raw/
└── processed/

docs/
├── architecture.md
├── api.md
├── mlops.md
└── operations.md

ops/
├── prometheus.yml
└── grafana/

migrations/
├── 0001_initial.py
└── 0002_platform_upgrade.py

scripts/
├── smoke_test.py
└── benchmark.py

CI/CD
├── testes multi-Python
├── build do wheel
├── build do container
├── CodeQL
├── pip-audit
├── Dependabot
└── pre-commit
```
