# API

A referência completa (schemas de entrada e saída) é gerada pelo FastAPI em `/docs` e `/redoc`, ou
exportada com `iris-ai export-openapi`.

## Autenticação

Quando `API_KEY` está definida, envie `X-API-Key: <chave>` em toda rota `/api`, exceto
`/api/v1/health`, `/api/v1/ready` e `/api/v1/metrics`. Sem a chave a resposta é 401. Fora de
`ENVIRONMENT=development` a chave é obrigatória para a aplicação iniciar.

A chave pode ser a mestra (`API_KEY`, papel `admin`) ou a de um usuário criado em `/api/v8/users`, com papel
`admin`, `operator` ou `reader`. Papel insuficiente responde 403. Veja
[Datasets, canário e papéis](datasets-canario-papeis.md#usuarios-e-papeis).

Toda resposta traz `X-Request-ID` (o do cliente, se for válido, ou um gerado).

## Rotas

| Método | Rota | Descrição |
|---|---|---|
| GET | `/healthz`, `/readyz` | sondas de vida e prontidão (banco) |
| GET | `/api/v1/health`, `/api/v1/ready` | saúde e prontidão (banco + modelo padrão registrado) |
| GET | `/api/v1/metrics` | métricas Prometheus |
| GET | `/api/v1/dataset/summary` | linhas, features e contagem por classe |
| POST | `/api/v1/dataset/seed` | carrega o Iris em `iris_records` (`?replace=true` recarrega) |
| GET/POST | `/api/v1/iris` | lista (filtro `class_name`) e cria registros |
| GET/PUT/DELETE | `/api/v1/iris/{id}` | lê, atualiza e remove |
| POST | `/api/v1/training/train` | treino síncrono; 422 para modelo desconhecido ou treino que falhou |
| GET | `/api/v1/training/runs` | histórico de treinos |
| POST | `/api/v1/predict` | predição unitária |
| POST | `/api/v1/predict/batch` | predição em lote (até `MAX_BATCH_SIZE`) |
| GET | `/api/v1/models`, `/api/v1/models/{nome}` | registry |
| GET | `/api/v2/platform` | versão, ambiente, autenticação, banco, backend de experimentos, extras instalados |
| GET | `/api/v2/data/quality` | gates de qualidade e perfil |
| GET | `/api/v2/features/manifest` | contrato das features (`?engineered=true` inclui áreas e razões) |
| GET | `/api/v2/models/leaderboard` | registros ordenados por uma métrica |
| POST | `/api/v2/models/{nome}/promote` | muda o estágio; `production` exige o release gate (409 se reprovar, `force` para forçar) |
| POST | `/api/v2/training/async` | treino em segundo plano (202 com `job_id`) |
| GET | `/api/v2/jobs`, `/api/v2/jobs/{id}` | estado dos jobs |
| GET | `/api/v2/experiments` | execuções registradas |
| GET | `/api/v2/monitoring/{modelo}` | resumo medido das últimas 500 predições |
| GET | `/api/v2/monitoring/drift/{modelo}` | só o relatório de drift |
| POST/GET | `/api/v2/monitoring/{modelo}/reports` | grava e lista snapshots |
| POST | `/api/v2/predictions/{id}/feedback` | registra a classe real de uma predição |
| POST | `/api/v2/explain` | Shapley exato da instância + importância global |
| GET | `/api/v3/meta` | versão, grupos de API montados, modelos disponíveis, ambiente |
| GET | `/api/v3/data/profile` | perfil por coluna |
| POST | `/api/v3/datasets/register`, GET `/api/v3/datasets` | catálogo de versões do dataset |
| GET | `/api/v4/data/anomalies` | Isolation Forest sobre o dataset |
| GET | `/api/v4/features/selection` | variância e SelectKBest (`anova` ou `mutual_info`) |
| POST | `/api/v4/pipelines/dry-run` | DAG de preparação sem treinar |
| POST | `/api/v4/pipelines/train`, GET `/api/v4/pipelines/runs` | pipeline completo e histórico |
| GET | `/api/v4/audit` | eventos de auditoria |
| GET | `/api/v4/governance/gate/{modelo}` | release gate de uma versão |
| GET | `/api/v4/analytics/correlations` | correlação (`pearson`, `spearman`, `kendall`) |
| GET | `/api/v4/deployment/manifest` | manifesto a partir do registro real |
| GET | `/api/v5/studio/frontier` | versões não dominadas em F1 x latência |
| POST | `/api/v5/studio/search` | busca TF-IDF no README e em `docs/` |
| POST | `/api/v5/studio/evaluate-predictions` | estimativa e IC 95% bootstrap |
| POST | `/api/v6/research/experiment-matrix` | validação cruzada repetida com o mesmo protocolo (até 60 treinos: modelos x dobras x repetições) |
| GET | `/api/v6/research/model-space` | modelos disponíveis e registrados |
| POST | `/api/v6/data/synthetic` | dados sintéticos (`make_classification`/`make_regression`) |
| POST | `/api/v6/modeling/robustness` | estabilidade do modelo registrado sob ruído |
| POST | `/api/v6/security/pii-scan` | detecção e mascaramento de e-mail, telefone, CPF e cartão |
| GET | `/api/v6/platform/sbom` | pacotes instalados no formato CycloneDX |
| POST | `/api/v6/governance/evidence-pack` | pacote de evidências de uma versão |
| POST | `/api/v7/labs/automl` | AutoML tabular no Iris (até 60 treinos: modelos x dobras) |
| POST | `/api/v7/labs/text` | classificador TF-IDF treinado com os textos enviados |
| POST | `/api/v7/labs/forecast` | previsão e backtest de uma série (mínimo 8 pontos) |
| GET | `/api/v7/labs/lineage-graph` | grafo dataset → treino → versão → estágio |
| POST | `/api/v8/datasets/profile` | perfil de um CSV (`text/csv`) sem gravar |
| POST | `/api/v8/datasets/upload?name=…&target=…` | perfil, validação e registro de um CSV no catálogo (201) |
| GET | `/api/v8/datasets`, `/api/v8/datasets/{nome}` | datasets enviados (versão mais recente) |
| POST | `/api/v8/predict` | predição com features nomeadas, para qualquer dataset; respeita o canário |
| POST | `/api/v8/explain` | Shapley exato com features nomeadas |
| POST/GET | `/api/v8/comparisons` | comparação por validação cruzada estratificada com IC (até 60 treinos: modelos x dobras); histórico |
| GET | `/api/v8/comparisons/{run_id}`, `…/html` | relatório JSON e HTML de uma execução |
| POST/GET | `/api/v8/canary` | abre um canário (201); lista |
| GET | `/api/v8/canary/{id}` | métricas por braço e decisão da regra; considera apenas predições atribuídas àquele canário |
| POST | `/api/v8/canary/{id}/evaluate`, `/rollback`, `/traffic` | aplica a regra, reverte, muda o percentual |
| GET | `/api/v8/users/me` | usuário e papel da chave enviada |
| GET/POST | `/api/v8/users` | lista e cria usuários (admin); a chave só aparece na criação |
| DELETE | `/api/v8/users/{id}` | revoga a chave de um usuário (admin) |

Cada linha nova em `prediction_logs` inclui `canary_id` e `canary_arm` quando a predição passa pela
divisão canário (`stable` ou `candidate`); predições normais deixam ambos nulos. Isso impede que chamadas
às rotas v1/v2 ou a outros canários contaminem as métricas de um experimento. A migração
`0007_canary_attribution` não inventa atribuições para logs antigos: um canário já ativo avalia somente
as novas observações identificadas.

## Exemplos

```http
POST /api/v1/predict
X-API-Key: <chave>
Content-Type: application/json

{"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2, "model_name": "xgboost"}
```

A resposta traz `prediction_id`, `model_version`, `predicted_class`, `confidence`, `probabilities`,
`latency_ms` e `request_id`. Com o `prediction_id`, envie o rótulo real:

```http
POST /api/v2/predictions/42/feedback
Content-Type: application/json

{"actual_class": "virginica"}
```

## Códigos de erro

- 401: API key ausente, errada ou revogada.
- 403: o papel do usuário não permite a operação.
- 404: modelo sem versão registrada, registro inexistente.
- 409: promoção barrada pelo release gate (o corpo traz o gate); canário em conflito; usuário repetido.
- 413: corpo acima de `MAX_REQUEST_BYTES` ou lote acima de `MAX_BATCH_SIZE`.
- 422: entrada inválida (schema ou regra de domínio).
- 429: rate limit (com `Retry-After`).
- 500: erro interno ou artefato que não passou na verificação de integridade.
