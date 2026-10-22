# MLOps

## Métricas de treino

Calculadas no conjunto de teste (20% estratificado, por padrão) do modelo final:

- `accuracy`, `balanced_accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, relatório por classe e
  matriz de confusão;
- médias da validação cruzada no treino e `generalization_gap` (treino menos validação);
- `brier_score` multiclasse (0 é perfeito, 2 é o pior caso);
- `latency_ms`: percentil 95 do tempo de `predict_proba` de uma única linha, medido no modelo treinado;
- `feature_importance`: queda de acurácia ao embaralhar cada feature no teste.

## Tuning

`tune_hyperparameters=true` (ou `--tune` na CLI):

- XGBoost usa Optuna (TPE) quando o extra `tuning` está instalado;
- sem Optuna, e para `random_forest`, `svm`, `logistic_regression` e `mlp`, usa `RandomizedSearchCV`
  com `optimization_trials` combinações;
- modelos sem espaço de busca recusam o pedido com erro explícito.

O método usado, os melhores parâmetros e a nota da busca ficam em `metrics.optimization`.

## Registry e estágios

```text
candidate → staging → production
                    ↘ archived
```

Cada versão guarda nome do arquivo, SHA-256, métricas, hash do dataset, hash de lineage, caminho do
model card e id da execução de treino. Promover uma versão a `production` arquiva a anterior e exige o
**release gate**: acurácia ≥ `GATE_MIN_ACCURACY`, F1 ≥ `GATE_MIN_F1` e `latency_ms` ≤
`GATE_MAX_LATENCY_MS`. Métrica ausente reprova. `force=true` ignora o gate e fica registrado na auditoria.

## Explicabilidade

`POST /api/v2/explain` calcula valores de Shapley exatos para a probabilidade da classe prevista: para
cada uma das 2⁴ = 16 coalizões de features, a probabilidade média com as features da coalizão fixas nos
valores da instância e as outras vindas de uma amostra de até 100 linhas do treino. As contribuições somam
exatamente `prediction - base_value`. A resposta inclui a importância global medida no treino.

## Monitoramento

`GET /api/v2/monitoring/{modelo}` usa as últimas 500 predições gravadas:

- confiança média, latência p50/p95 e distribuição de classes previstas;
- acurácia sobre as predições que receberam feedback (`/api/v2/predictions/{id}/feedback`);
- drift por PSI de cada feature contra o Iris (`stable`, `warning` a partir de `DRIFT_PSI_WARNING`,
  `drift` a partir de `DRIFT_PSI_THRESHOLD`), a partir de 30 predições;
- SLOs: `p95_latency_ms` ≤ `SLO_P95_LATENCY_MS`, `mean_confidence` ≥ `SLO_MIN_MEAN_CONFIDENCE` e
  `accuracy_when_labeled` ≥ `GATE_MIN_ACCURACY` (SLO sem dado fica `unknown`);
- nota de saúde de 0 a 100: -30 se a qualidade do dataset falhar, -15 para drift `warning`, -35 para
  `drift`, -15 por SLO violado. Cada desconto aparece em `health.penalties`.

`POST /api/v2/monitoring/{modelo}/reports` grava o resumo em `monitoring_reports` e publica os gauges
`iris_model_health_score` e `iris_model_drift_psi_max`.

## Experimentos

Cada treino gera uma linha em `experiment_runs`. Com `MLFLOW_TRACKING_URI` definido e o extra `tracking`
instalado, parâmetros e métricas numéricas também vão para o MLflow (`backend = mlflow`).
