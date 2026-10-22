# Operação

## Configuração

Todas as opções são variáveis de ambiente (ou `.env`); a lista comentada está em `.env.example`. As mais
importantes:

| Variável | Padrão | Uso |
|---|---|---|
| `ENVIRONMENT` | `development` | qualquer outro valor exige `API_KEY` |
| `API_KEY` | vazio | chave exigida no header `X-API-Key` |
| `ARTIFACT_SIGNING_KEY` | vazio | assina e exige assinatura HMAC dos artefatos |
| `DATABASE_URL` | `sqlite:///./data/iris.db` | URL SQLAlchemy; testados SQLite e PostgreSQL (`postgresql+psycopg://…`, com `pip install "psycopg[binary]>=3.2,<4"`) |
| `MODEL_DIR`, `ARTIFACT_DIR`, `DATA_DIR` | `models`, `artifacts`, `data` | relativos ao diretório de trabalho |
| `MAX_REQUEST_BYTES`, `MAX_BATCH_SIZE` | 2 MB, 500 | limites de entrada |
| `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS` | 300, 60 | rate limit por IP |

## Migrações

O schema é criado automaticamente na inicialização. Para operar à mão: `alembic upgrade head`,
`alembic history`, `alembic downgrade -1` (a URL vem de `DATABASE_URL`). A 8.1 acrescenta a revisão
`0005_users_canary` (`api_users`, `canary_deployments` e `prediction_logs.model_version`) e a
`0006_unique_active_canary` (índice único parcial: um canário ativo por modelo).

Migrações, SQL offline (`alembic upgrade --sql`) e a suíte inteira passam também no PostgreSQL 18:
`TEST_DATABASE_URL=postgresql+psycopg://usuario:senha@host:5432/banco_vazio pytest`.

## Usuários

Gere chaves por pessoa ou sistema em vez de distribuir a `API_KEY`: `iris-ai user-create ana --role operator`
(ou `POST /api/v8/users` com a chave mestra). A chave aparece só na criação; o banco guarda o hash.
Revogue com `DELETE /api/v8/users/{id}`. CSVs enviados ficam em `DATA_DIR/uploads` e relatórios de
comparação em `ARTIFACT_DIR/comparisons`: inclua os dois no backup.

## Docker e Compose

```bash
cp .env.example .env   # defina API_KEY e GRAFANA_ADMIN_PASSWORD
docker compose up --build -d
```

A imagem roda como usuário sem privilégios (UID 10001), com `ENVIRONMENT=production`, e guarda banco,
modelos e artefatos no volume `/app/state`. Prometheus coleta `/api/v1/metrics`; o Grafana carrega o
painel `ops/grafana/dashboards/iris-ai.json`.

## Kubernetes (Helm) e Terraform

O chart `infra/helm/iris-ai-platform` cria Deployment (1 réplica, estratégia `Recreate`,
`securityContext` restritivo, raiz somente leitura), Service e PersistentVolumeClaim. A `API_KEY` vem de
um Secret existente:

```bash
kubectl create secret generic iris-ai-secrets --from-literal=API_KEY=<valor>
helm install iris-ai infra/helm/iris-ai-platform --set image.repository=<registro>/iris-ai-platform
```

`infra/terraform/main.tf` instala o mesmo chart com o provider Helm (`terraform init && terraform apply
-var image_repository=<registro>/iris-ai-platform`).

## O que não foi verificado

O Dockerfile, o Compose, o chart Helm e o Terraform foram revisados, mas **não executados** no ambiente
em que a versão 8.0 foi preparada (sem Docker, Helm, kubectl nem Terraform instalados). Rode
`docker compose config`, `helm lint`/`helm template` e `terraform validate` antes de usar.

## Checklist de produção

- `ENVIRONMENT=production`, `API_KEY` forte e `ARTIFACT_SIGNING_KEY` definidas;
- API atrás de um proxy com TLS; `CORS_ORIGINS` restrito;
- backup do volume de estado (banco e `models/`);
- alertas sobre `iris_model_health_score` e `iris_model_drift_psi_max` (alimentados por snapshots
  periódicos em `POST /api/v2/monitoring/{modelo}/reports`).
