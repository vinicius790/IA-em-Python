# Datasets, comparação, canário e papéis (8.1)

A versão 8.1 tira a plataforma do Iris: qualquer CSV de classificação pode ser enviado, perfilado,
treinado, comparado e servido. Também traz deploy canário com decisão automática e usuários com papéis.
As rotas novas ficam em `/api/v8` e todas passam pela mesma autenticação das demais.

## Datasets enviados

### Upload e perfil

O CSV vai no corpo da requisição (`Content-Type: text/csv`, UTF-8, com cabeçalho), limitado por
`MAX_REQUEST_BYTES`:

```bash
curl -X POST "http://127.0.0.1:8000/api/v8/datasets/upload?name=vinhos" \
  -H "X-API-Key: $API_KEY" -H "Content-Type: text/csv" --data-binary @vinhos.csv
```

Pela CLI: `iris-ai dataset-add vinhos.csv --name vinhos [--target tipo]`.

O perfil (`app/data_catalog/csv_profile.py`, só biblioteca padrão) devolve, por coluna, o tipo
(`numeric`, `categorical` ou `empty`), nulos e taxa de nulos, cardinalidade, estatísticas (mínimo, máximo,
média, desvio, mediana) das numéricas e os valores mais frequentes das categóricas. Contam como nulo
(sem diferenciar maiúsculas): vazio, `NA`, `N/A`, `NaN`, `null`, `None` e `?`. É numérica a coluna em que
todo valor não nulo está na notação decimal comum (`12`, `-3.5`, `.5`, `2e3`); `1_000`, `inf` ou dígitos não
ASCII tornam a coluna categórica. O treino lê o arquivo com o mesmo leitor e as mesmas regras do perfil.

**Alvo.** Com `target`, a coluna indicada é usada. Sem ele, o alvo é a última coluna se ela parecer de
classes (de 2 a 50 valores distintos, sem nulos, texto ou inteiros); senão, a coluna categórica de menor
cardinalidade que cumpra a mesma regra. **Features** são as colunas numéricas que não são o alvo; colunas de
texto ficam de fora do treino (e aparecem no perfil).

**Recusa (422, com o motivo).** Nome fora de `[a-z0-9][a-z0-9_-]{0,63}` ou igual a `iris`; arquivo vazio,
sem linhas, com cabeçalho repetido ou vazio, CSV malformado, linha com número errado de campos, texto que não
é UTF-8; menos de 30 linhas; alvo inexistente, com nulos, com uma só classe ou com uma classe de uma linha só;
nenhuma feature numérica; feature com nulos ou mais de 1% de linhas repetidas em features + alvo (os mesmos
gates de qualidade do treino). `POST /api/v8/datasets/profile` faz o mesmo perfil sem gravar nada e lista os
problemas.

**Catálogo.** Cada versão fica em `DATA_DIR/uploads/<nome>/<sha256[:16]>.csv` e entra no catálogo
(`DATA_DIR/catalog.json`) com SHA-256, linhas, perfil, alvo e features. Reenviar o mesmo nome registra uma
nova versão sem apagar a anterior: treinos novos usam a mais recente, e cada modelo guarda no registry
(`metadata.dataset_sha256`) a versão em que foi treinado, que é a referência do seu drift e da robustez.
O catálogo é regravado sob uma trava de arquivo (a mesma do registry), então vários processos (workers
do uvicorn, a CLI) podem registrar ao mesmo tempo no mesmo `DATA_DIR`. As respostas da API
(`/api/v8/datasets…` e `/api/v3/datasets…`) não trazem o caminho do arquivo no servidor.

### Treino parametrizado pelo dataset

`POST /api/v1/training/train` e `POST /api/v2/training/async` aceitam `"dataset": "vinhos"`
(`iris-ai train --dataset vinhos` na CLI). O modelo é registrado como `<dataset>.<modelo>`
(`vinhos.random_forest`), com versões próprias, para não misturar dados diferentes no mesmo nome. Gates de
qualidade, métricas, model card, lineage e artefato seguem o mesmo fluxo do Iris; o artefato guarda as
features, o alvo e o nome do dataset.

### Predição e explicação com features nomeadas

```http
POST /api/v8/predict
{"model_name": "vinhos.random_forest", "features": {"acidez": 3.2, "acucar": 10.5, "teor": 10.0}}
```

As features são reordenadas na ordem do modelo; faltando ou sobrando alguma, a resposta é 422 com a lista
(e também para `NaN` ou `Infinity`).
`POST /api/v8/explain` dá o Shapley exato (até 10 features). As rotas da v1/v2 continuam com o formato do
Iris (4 features); usadas com um modelo de outro tamanho respondem 422. O monitoramento
(`/api/v2/monitoring/{modelo}`) mede o drift contra o dataset em que o modelo foi treinado.

## Comparação de modelos

```http
POST /api/v8/comparisons
{"models": ["random_forest", "svm", "knn"], "dataset": "vinhos", "folds": 5, "confidence": 0.95}
```

CLI: `iris-ai compare random_forest svm knn --dataset vinhos --folds 5`.

- Todos os modelos são avaliados nas **mesmas** dobras de uma validação cruzada estratificada (semente
  `RANDOM_STATE`, ou `random_state` no corpo).
- Para F1 macro e acurácia: média, desvio amostral e intervalo de confiança t de Student da média das
  dobras (90%, 95% ou 99%).
- Ranking por F1 macro. Para cada modelo, o intervalo da **diferença pareada** de F1 para o melhor (dobra a
  dobra); `differs_from_best` é verdadeiro quando esse intervalo não contém zero.
- Cada execução recebe um `run_id` e grava `ARTIFACT_DIR/comparisons/<run_id>.json` e `.html`, consultáveis
  em `GET /api/v8/comparisons/{run_id}` e `GET /api/v8/comparisons/{run_id}/html`.

Com poucas dobras o intervalo é largo, e as dobras de uma validação cruzada não são independentes (os
treinos se sobrepõem), então o intervalo tende a ser otimista. Use-o para ordenar e descartar, não como
teste formal.

## Deploy canário

1. Tenha uma versão em `production` (a estável) e treine uma nova versão do mesmo modelo (a candidata).
2. Abra o canário:

   ```http
   POST /api/v8/canary
   {"model_name": "vinhos.random_forest", "candidate_version": 3, "traffic_percent": 10,
    "rule": {"min_requests": 50, "max_confidence_drop": 0.05, "max_latency_ratio": 1.5,
             "max_accuracy_drop": 0.02, "min_labeled": 20}}
   ```

3. `POST /api/v8/predict` divide o tráfego: o braço sai do hash de `routing_key` (ou do `X-Request-ID`), então
   o mesmo cliente cai sempre na mesma versão. A resposta traz `arm`, `model_version` e `canary_id`. As rotas
   v1/v2 seguem servindo a versão estável.
4. Cada predição registra versão, ID do canário e braço (`stable` ou `candidate`). `GET /api/v8/canary/{id}`
   mostra as métricas **separadas** de cada braço (somente observações daquele experimento: predições,
   confiança média, latência p95 e acurácia nas predições com feedback) e o que a regra decidiria agora.
   Tráfego das rotas v1/v2 e de outros canários não entra nessa avaliação.

**Regra (`app/deployment/canary.py`).** Antes de `min_requests` predições da candidata a decisão é
esperar. Depois disso a candidata é **revertida** se a confiança média cair mais que `max_confidence_drop`,
se o p95 de latência passar de `max_latency_ratio` vezes o da estável, ou (com pelo menos `min_labeled`
predições rotuladas em cada braço, via `/api/v2/predictions/{id}/feedback`) se a acurácia cair mais que
`max_accuracy_drop`; senão é **promovida**. As métricas de cada braço usam as 500 predições mais recentes,
então `min_requests` e `min_labeled` vão de 1 a 500. Como antes de `min_requests` a regra só pode esperar e
depois disso sempre decide, cada predição da candidata faz só uma contagem limitada (até `min_requests`
ids) e a regra completa roda uma vez, quando o mínimo é atingido; sob demanda, há
`POST /api/v8/canary/{id}/evaluate`. O canário é fechado por um `UPDATE` condicional:
se duas decisões chegam juntas (predições simultâneas, reversão manual durante a promoção), só a primeira
mexe no registry; a outra recebe 409 (uma predição apenas devolve o estado já decidido).

**Atribuição do experimento.** A migração `0007_canary_attribution` acrescenta `canary_id` e
`canary_arm` ao log e um índice composto para recuperar as observações por experimento. Os dois campos são
gravados juntos apenas por `/api/v8/predict` enquanto há um canário ativo. Logs anteriores à migração não
são retroativamente classificados, pois não é possível provar qual requisição pertenceu ao experimento;
canários já ativos usam somente as novas observações identificadas.

- **Promoção**: a candidata vai para `production` pelo mesmo caminho da promoção manual, então o release gate
  vale; se ele reprovar, o canário é revertido.
- **Reversão**: a candidata vai para `archived` e a estável volta a receber todo o tráfego. Também há
  reversão manual (`POST /api/v8/canary/{id}/rollback`) e mudança de percentual
  (`POST /api/v8/canary/{id}/traffic`).
- Um canário ativo por modelo, garantido pelo banco (índice único parcial em `status = 'active'`): abrir
  outro com um ativo, mesmo em duas requisições simultâneas, sem versão em produção ou com candidata
  arquivada dá 409.
- Tudo fica na tabela `canary_deployments` e na auditoria (`canary_started`, `canary_promoted`,
  `canary_rolled_back`, `canary_traffic`).

CLI: `iris-ai canary-start vinhos.random_forest 3 --traffic 10` e `iris-ai canary-status 1 [--evaluate]`.

## Usuários e papéis

A `API_KEY` continua sendo a chave mestra, com papel `admin`. Além dela, cada usuário de `api_users` tem a
sua chave e um papel:

| Papel | Pode |
|---|---|
| `reader` | todo `GET`, e as rotas `POST` que não alteram estado: predição, lote, explicação, busca na documentação, avaliação de predições e varredura de PII |
| `operator` | tudo de `reader`, mais treino, upload, comparação, canário, promoção, feedback, CRUD de registros |
| `admin` | tudo, inclusive `GET/POST /api/v8/users` e `DELETE /api/v8/users/{id}` |

- `POST /api/v8/users` (`{"name": "ana", "role": "operator"}`) devolve a chave **uma única vez**; o banco
  guarda só o SHA-256 dela. Na CLI: `iris-ai user-create ana --role operator` e `iris-ai users`.
- `DELETE /api/v8/users/{id}` revoga a chave (o registro fica, para a auditoria continuar legível).
- `GET /api/v8/users/me` diz quem a chave identifica.
- Chave errada ou revogada: 401. Papel insuficiente: 403, com o papel exigido.
- A auditoria registra o usuário como `actor` e o papel em `payload.actor_role`. Ações da CLI aparecem como
  `api` (ou `cli`).
- Em `ENVIRONMENT=development` sem `API_KEY`, a API continua aberta e quem chama é tratado como `admin`.

## Migração

A revisão `0005_users_canary` cria `api_users` e `canary_deployments` e acrescenta `model_version` a
`prediction_logs`. A `0006_unique_active_canary` cria o índice único parcial de um canário ativo por modelo
(SQLite e PostgreSQL; se o banco já tiver dois ativos para o mesmo modelo, fecha o mais antigo). As duas
são idempotentes para bancos adotados do antigo `create_all()`, têm `downgrade` e geram SQL no modo
offline (`alembic upgrade --sql`).
