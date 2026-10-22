# Changelog

As versões estão em ordem decrescente. A versão do pacote tem uma única fonte: `app/__init__.py`.

## 8.2.0 - 2026-10-16

Pipeline de Explicabilidade (XAI), Validação Cruzada K-Fold com Intervalos de Confiança, Detecção Estatística Bicaudal de Data Drift (PSI e KS-test), Exportação de Relatório de Auditoria de Modelo (Markdown/Mermaid e JSON), CLI Avançada, Dockerfile Multi-Stage e CI/CD atualizado.

### Adicionado

- **Pipeline de Explicabilidade (XAI)** (`app/modeling/explainability.py`):
  - Importância nativa de features para modelos baseados em árvores e lineares (`extract_model_importance`).
  - Importância global por permutação pura (`pure_permutation_importance` e `run_permutation_importance`) calculando média e desvio padrão sem dependências C pesadas.
  - Decomposição local aditiva de predições via Shapley Exato (para até 10 features) e Kernel SHAP amostrado via WLS (`compute_kernel_shap`), garantindo reconciliação aditiva estrita $\sum \phi_i = f(x) - \mathbb{E}[f(X)]$.
  - Passos acumulativos de gráfico Waterfall e resumo textual formatado.
- **Validação Cruzada K-Fold Robusta** (`app/modeling/cross_validation.py`):
  - Motor de validação cruzada $K$-Fold e Estratificada out-of-fold com instâncias clonadas e isoladas do estimador.
  - Métricas agregadas por dobra e consolidadas: Acurácia, Acurácia Balanceada, F1-Macro, Precisão Macro e Revocação Macro.
  - Cálculo de intervalos de confiança bilaterais a 95% via distribuição t de Student.
  - Matriz de confusão agregada out-of-fold.
- **Detecção Estatística de Data Drift (Dual-Engine)** (`app/monitoring/drift.py`):
  - Integração do teste bicaudal não-paramétrico de Kolmogorov-Smirnov (`ks_test`) ao cálculo existente de PSI (`psi`).
  - Análise estatística multivariada completa (`statistical_drift_analysis`) classificando cada atributo em `stable`, `warning` ou `drift`.
- **Relatório de Auditoria e Governança Técnica (Model Audit Report)** (`app/governance/model_audit.py`):
  - Construtor do relatório técnico consolidado (`build_model_audit_report`) integrando gates de qualidade de dados, K-Fold, XAI, Data Drift e checklist de conformidade regulatória.
  - Veredito executivo automatizado (`APPROVED`, `WARNING`, `REJECTED`).
  - Exportação estruturada (`export_audit_report`) em JSON e em Markdown formatado com diagramas Mermaid embutidos.
- **CLI Avançada** (`app/cli.py`):
  - Comando `iris-ai explain-pipeline`: executa e exibe ranking de explicabilidade e importância por permutação.
  - Comando `iris-ai kfold`: executa validação cruzada K-Fold exibindo médias, desvios e intervalos de confiança a 95%.
  - Comando `iris-ai drift-check`: avalia Data Drift estatístico combinando PSI e KS-test.
  - Comando `iris-ai audit-export`: gera e exporta o relatório de auditoria e conformidade técnica do modelo em formato Markdown ou JSON.
- **Dockerfile Multi-Stage**:
  - Estágio `builder` com compilação e isolamento de dependências em `/opt/venv`.
  - Estágio `runtime` minimalista baseado em `python:3.13-slim`, sem ferramentas de compilação, com usuário não-privilegiado `app:app` (UID/GID 10001), healthcheck nativo e volumes persistentes.
- **CI/CD no GitHub Actions** (`.github/workflows/ci.yml`):
  - Pipeline com matriz Python 3.12 e 3.13, verificação estática com Ruff, suíte de testes com Pytest, validação de ponta a ponta dos novos comandos da CLI, smoke tests na API e validação de build da imagem Docker multi-stage.
- **Suíte de Testes Automatizados**: `tests/test_audit_and_advanced_pipelines.py` cobrindo 100% dos novos motores e comandos da CLI.

## 8.1.0 - 2026-10-03

Datasets além do Iris, comparação de modelos, deploy canário e usuários com papéis. Nova API `/api/v8`;
as rotas anteriores continuam iguais.

### Robustez MLOps

- Predições canário agora persistem `canary_id` e `canary_arm`; relatórios e o gatilho automático de decisão
  agregam somente observações daquele experimento, sem misturar tráfego normal das rotas v1/v2 ou de outros
  canários. A migração `0007_canary_attribution` adiciona colunas nulas para preservar logs antigos e índice
  composto para as consultas por canário/braço. Logs antigos não recebem atribuição inferida.

### Adicionado

- Upload de CSV (`POST /api/v8/datasets/upload`, `iris-ai dataset-add`) com perfil automático em biblioteca
  padrão (tipos, nulos, estatísticas, alvo inferido ou explícito), validação com motivo da recusa (inclusive
  pelos mesmos gates de qualidade do treino) e registro no catálogo com SHA-256, um arquivo por versão.
  `POST /api/v8/datasets/profile` faz o perfil sem gravar.
- Treino parametrizado pelo dataset (`"dataset"` no treino síncrono e assíncrono, `--dataset` na CLI): o
  modelo é registrado como `<dataset>.<modelo>`; gates de qualidade, model card e artefato usam as features
  e o alvo do dataset, lido com as mesmas regras do perfil. Predição e explicação com features nomeadas em
  `/api/v8/predict` e `/api/v8/explain`; o drift é medido contra a versão do dataset em que o modelo foi
  treinado (`metadata.dataset_sha256` no registry).
- Comparação de modelos (`POST /api/v8/comparisons`, `iris-ai compare`): validação cruzada estratificada
  com as mesmas dobras para todos, intervalo de confiança t por métrica e da diferença pareada para o
  melhor, relatório JSON e HTML por execução.
- Deploy canário (`/api/v8/canary`, `iris-ai canary-start`/`canary-status`): divisão de tráfego
  determinística por chave, métricas separadas por braço, promoção (pelo release gate) ou reversão
  automática por regra, reversão manual e ajuste de percentual. O fechamento é um `UPDATE` condicional:
  decisões simultâneas não promovem e arquivam a mesma candidata. Um canário ativo por modelo, garantido
  por índice único parcial (aberturas simultâneas: uma vence, a outra recebe 409). A regra completa roda
  uma vez, quando a candidata atinge `min_requests`; antes disso cada predição faz só uma contagem limitada.
- Usuários e papéis (`admin`, `operator`, `reader`) sobre a API key atual, que continua sendo a chave
  mestra: `/api/v8/users`, `iris-ai user-create`/`users`. Chaves guardadas como SHA-256, revogáveis; 403
  para papel insuficiente; a auditoria registra o usuário e o papel.
- Migrações `0005_users_canary` (`api_users`, `canary_deployments`, `prediction_logs.model_version`) e
  `0006_unique_active_canary` (um canário ativo por modelo), ambas com `downgrade` e modo offline.
- Testes: `tests/test_v8.py` (API, CLI e migração) e `tests/test_stdlib_units.py` (regras puras, roda com
  `python -m unittest` sem dependências).

### Alterado

- `validate_training_frame`, `train_model` e `create_model_card` aceitam features, alvo e nome do dataset;
  sem eles o comportamento do Iris é o mesmo.
- Predição com número errado de features responde 422 (antes era 500).
- Corpo com `NaN` ou `Infinity` em campo numérico responde 422 em qualquer rota (antes o handler padrão
  ecoava o valor, a resposta não era JSON válido e virava 500).
- `POST /api/v6/modeling/robustness` usa o dataset em que o modelo foi treinado.
- Comparação, matriz de experimentos e AutoML aceitam no máximo 60 treinos de validação cruzada por
  requisição (422 acima disso).
- `/api/v3/datasets` deixa de devolver o caminho do arquivo no servidor (as rotas de datasets da v8
  também não o devolvem).
- O catálogo de datasets é gravado sob a mesma trava de arquivo do registry (seguro entre processos).
- `TEST_DATABASE_URL` roda a suíte contra outro banco; migrações, SQL offline e testes verificados no
  PostgreSQL 18 com `psycopg` 3.

## 8.0.0 - 2026-10-02

Revisão geral: segurança, correção de bugs e remoção do que era fachada. Há mudanças incompatíveis.

### Segurança

- Removida a rota `POST /api/v2/data/query`, que executava SQL arbitrário no DuckDB e permitia ler arquivos do servidor.
- Uma única dependência de autenticação (`app/security.py`, comparação em tempo constante) aplicada a todos os routers `/api`. Só as sondas de saúde, as métricas e o console estático ficam públicos.
- Fora de `ENVIRONMENT=development` a aplicação não inicia sem `API_KEY`.
- Removida a rota `/api/v6/foundry/manifest`, que fazia hash do repositório inteiro (incluindo `.env`).
- O limite de corpo passa a contar os bytes recebidos, não só o header `Content-Length`.
- `X-Request-ID` do cliente só é aceito se for curto e sem caracteres especiais.
- O rate limiter descarta baldes vencidos; o label `path` das métricas usa o template da rota.
- Artefatos só são desserializados depois da verificação de integridade; assinatura HMAC opcional com `ARTIFACT_SIGNING_KEY`.
- O registry guarda só o nome do arquivo do artefato, sempre dentro do diretório de modelos.

### Corrigido

- `POST /api/v5/studio/search` respondia 500 (`Path` não importado) e varria o repositório inteiro.
- `python -m app.cli` não enxergava os comandos definidos depois do `if __name__ == "__main__"`.
- `POST /api/v7/labs/forecast` respondia 500 com séries curtas aceitas pelo schema; o backtest ignorava `window`.
- Entradas inválidas nas rotas v4 a v7 respondiam 500; agora respondem 422.
- `--tune` sem Optuna não fazia tuning nenhum: agora usa RandomizedSearchCV e registra o método; modelo sem espaço de busca é erro explícito.
- `/api/v1/predict/batch` fazia uma predição por item; agora faz uma predição vetorizada por modelo.
- O release gate passava quando `latency_ms` não existia; a latência agora é medida no treino e métrica ausente reprova.
- O manifesto de implantação apontava para um arquivo que não existia.
- A numeração de versões do registry podia repetir sob concorrência (agora há lock e `max + 1`).
- Jobs assíncronos de treino que falhavam apareciam como `completed`.
- O console fazia `GET` numa rota `POST`, não enviava a API key e usava `innerHTML` com dados da API.
- Os testes gravavam banco e modelos dentro do repositório.

### Alterado

- `POST /api/v2/explain` devolve valores de Shapley exatos da instância, além da importância global medida no treino.
- Monitoramento real em `GET /api/v2/monitoring/{modelo}`: confiança, latência, acurácia rotulada, drift, SLOs e nota de saúde calculados a partir das predições gravadas. `POST /api/v2/predictions/{id}/feedback` registra o rótulo real.
- Promoção para `production` exige passar no release gate (ou `force`), com auditoria.
- Auditoria e execuções de pipeline gravadas no banco (`audit_events`, `pipeline_runs`).
- Alembic é o único caminho de schema (`init_db()` roda `upgrade head`); nova migração `0004`.
- Grafo de linhagem (`/api/v7/labs/lineage-graph`) montado a partir do registry.
- Dependências declaradas só no `pyproject.toml`; extras reduzidos a `tuning`, `boosters`, `tracking`, `docs` e `dev`.
- Infra: chart Helm com PVC, Secret e `securityContext`; Terraform que instala o chart; Compose exige `API_KEY`.

### Removido

- Stubs e código sem uso: adapters (Ray, Dask, ONNX, S3, Redis), "agente" por palavras-chave, copilot com provedor falso, guardrails de prompt, RBAC, plugins, roteador de tráfego, micro-batching, circuit breaker, planos de rollout, workspaces, feature store local, idempotência, event bus, telemetria OpenTelemetry vazia, blueprints, integrações Airflow/Prefect/Evidently/DVC/Feast, definição gRPC sem servidor.
- Rotas que devolviam valores fixos (`/api/v3/health/score`, `/api/v3/reports/demo`, grafo fixo da v7, entre outras) e rotas duplicadas.
- Arquivos de compatibilidade `Main.py`, `predict.py` e `neural_network.py` da raiz, `VERSION`, `requirements.txt`, `docker-compose.full.yml`, manifestos Kubernetes duplicados do chart e workflows de CI que não se aplicavam.
- Modelo `calibrated_svm` (duplicava `svm`, que passou a usar calibração no lugar de `SVC(probability=True)`, obsoleto no scikit-learn 1.9).

## 7.0.0

- AutoML tabular (classificação e regressão), classificador de texto TF-IDF, baselines de séries temporais e API v7.

## 6.0.0

- Avaliação estatística (bootstrap), fronteira de Pareto, busca na documentação e API v5/v6.

## 5.1.0

- Pipeline de treino executável e estatísticas do dataset.

## 5.0.0

- Seleção de features, detecção de anomalias, trilha de auditoria e API v4.

## 4.0.0

- Orquestração por DAG, catálogo e contratos de datasets, engenharia de features, release gates e API v3.

## 3.0.0 - 2026-09-30

- Catálogo de modelos com ensembles e boosters opcionais, tuning, tracking de experimentos, perfil e gates de qualidade, drift PSI, lineage, model cards, registry com versões e estágios, API v2 com jobs assíncronos, Prometheus e Grafana, migrações Alembic.

## 2.0.0

- Arquitetura FastAPI + SQLAlchemy + scikit-learn, registro local de modelos, métricas Prometheus, CI e testes.
