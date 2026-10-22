# Arquitetura de Dados e Plataforma MLOps

Este documento detalha a arquitetura de dados, o ciclo de vida completo dos modelos de Machine Learning, a esteira de detecção de *drift*, a orquestração de *pipelines* e a topologia de governança da plataforma **Iris AI Platform** (`IA-em-Python`).

---

## 1. Arquitetura de Dados e Fluxo Geral da Informação

A plataforma opera com uma arquitetura híbrida de persistência:
1. **Metadados e Estado Operacional:** Banco de dados relacional (SQLite em desenvolvimento / PostgreSQL em produção gerenciado via Alembic).
2. **Artefatos e Model Store:** Armazenamento seguro de objetos/arquivos (`MODEL_DIR` e `ARTIFACT_DIR`) com locks de arquivo, verificação de integridade via checksum SHA-256 e assinatura criptográfica HMAC.

```mermaid
flowchart TD
    subgraph DataIngestion["1. Camada de Ingestão de Dados"]
        RawData["Dataset Iris (sklearn) / CSVs Externos"]
        CatalogIngest["Catalog Ingestion & Schema Contracts<br/><i>(data_catalog/contracts.py)</i>"]
        QualityGate["Quality Checks & PII Redaction<br/><i>(quality.py & governance/pii.py)</i>"]
    end

    subgraph StorageLayer["2. Camada de Persistência Híbrida"]
        RelationalDb[("Banco Relacional (Alembic)<br/>iris_records, training_runs, prediction_logs,<br/>canary_deployments, audit_events, pipeline_runs")]
        ArtifactStore[("Model & Artifact Store<br/>catalog.json (FileLock), *.joblib,<br/>model_cards/*.json, reports/*.html")]
    end

    subgraph MLOpsCore["3. Motor MLOps e Ciclo de Vida"]
        TrainingEngine["ML Training & Optuna Tuning<br/><i>(ml.py & optimization.py)</i>"]
        RegistryManager["Model Registry & Release Gates<br/><i>(registry.py & deployment/gates.py)</i>"]
        CanaryRouter["Canary Deployment & Routing<br/><i>(deployment/canary.py)</i>"]
    end

    subgraph ServingObservability["4. Inferência e Observabilidade"]
        InferenceAPI["FastAPI Serving Engine<br/><i>(services.py & api_v1...api_v8)</i>"]
        DriftMonitor["Drift Engine (PSI) & Health Scoring<br/><i>(monitoring/drift.py & health.py)</i>"]
        MetricsPrometheus["Métricas Prometheus & MLflow<br/><i>(iris_model_health_score, PSI max)</i>"]
    end

    RawData --> CatalogIngest
    CatalogIngest --> QualityGate
    QualityGate -->|"Grava dados auditados"| RelationalDb
    QualityGate -->|"Fornece dataset validado"| TrainingEngine
    
    TrainingEngine -->|"Registra execução & métricas"| RelationalDb
    TrainingEngine -->|"Salva artefato binário & model card"| ArtifactStore
    TrainingEngine --> RegistryManager
    
    RegistryManager -->|"Avalia Release Gate"| CanaryRouter
    CanaryRouter --> InferenceAPI
    InferenceAPI -->|"Registra predições e latências"| RelationalDb
    InferenceAPI -->|"Gera logs de inferência"| DriftMonitor
    
    DriftMonitor -->|"Compara vs dataset de referência"| RelationalDb
    DriftMonitor -->|"Exporta alertas e score de saúde"| MetricsPrometheus
```

---

## 2. Diagrama Entidade-Relacionamento dos Metadados de MLOps

O banco relacional mantém o rastreamento ponto a ponto de cada dataset, treinamento, predição, canário e ação de auditoria:

```mermaid
erDiagram
    IRIS_RECORD ||--o{ TRAINING_RUN : "fornece dados via hash"
    TRAINING_RUN ||--o{ PIPELINE_RUN : "executado por"
    TRAINING_RUN ||--o{ EXPERIMENT_RUN : "gera experimentos"
    CANARY_DEPLOYMENT ||--o{ PREDICTION_LOG : "roteia tráfego"
    API_USER ||--o{ AUDIT_EVENT : "autor da ação"
    
    IRIS_RECORD {
        int id PK "Autoincrement"
        float sepal_length "Comprimento da sépala"
        float sepal_width "Largura da sépala"
        float petal_length "Comprimento da pétala"
        float petal_width "Largura da pétala"
        string class_name "Classe alvo (setosa, versicolor, virginica)"
        string source "Origem (sklearn, csv_catalog)"
        datetime created_at "Data de ingestão"
    }

    TRAINING_RUN {
        int id PK "Identificador da execução"
        string model_name "Nome do algoritmo (xgboost, random_forest, etc)"
        string status "running, completed, failed"
        json metrics "Acurácia, F1, Brier, latência p95, feature importance"
        json parameters "Hiperparâmetros usados"
        text artifact_path "Caminho do artefato .joblib"
        string dataset_hash "Hash SHA-256 do dataset utilizado"
        string lineage_hash "Hash do grafo de linhagem"
        text model_card_path "Caminho do Model Card"
        datetime created_at "Data de execução"
    }

    PREDICTION_LOG {
        int id PK "Identificador da predição"
        string model_name "Modelo acionado"
        int model_version "Versão do modelo servida"
        int canary_id "ID do canário ativo (se aplicável)"
        string canary_arm "Braço do canário: stable ou candidate"
        string predicted_class "Classe inferida"
        float confidence "Score de confiança / probabilidade máxima"
        json features "Valores das features de entrada"
        float latency_ms "Latência de inferência em ms"
        string request_id "Identificador de rastreamento HTTP"
        string actual_class "Ground truth recebido a posteriori via feedback"
        datetime created_at "Momento da inferência"
    }

    CANARY_DEPLOYMENT {
        int id PK "Identificador do deploy canário"
        string model_name "Nome do modelo avaliado"
        int stable_version "Versão em produção de referência"
        int candidate_version "Versão candidata em teste de tráfego"
        float traffic_percent "Percentual de tráfego (0 a 100%)"
        string status "active, promoted, rolled_back"
        json rule "Regras de canário (tempo, limite de erros)"
        int baseline_prediction_id "Predição inicial do teste"
        string created_by "Autor do deploy canário"
        datetime created_at "Data de abertura"
        datetime decided_at "Data de conclusão"
    }

    MONITORING_REPORT {
        int id PK "Identificador do relatório"
        string model_name "Modelo avaliado"
        string report_type "drift, health, slo"
        string status "stable, warning, drift"
        json payload "Métricas completas de PSI e penalidades"
        datetime created_at "Data de emissão"
    }

    AUDIT_EVENT {
        int id PK "Identificador do evento"
        string action "Ação auditada (promote_model, force_gate, etc)"
        string actor "Usuário ou processo responsável"
        json payload "Detalhes contextuais da operação"
        datetime created_at "Data do evento"
    }

    PIPELINE_RUN {
        int id PK "Identificador do pipeline"
        string pipeline_name "Nome da DAG (iris-training)"
        string status "running, completed, blocked, failed"
        string model_name "Modelo treinado no pipeline"
        json payload "Relatório completo de execução"
        datetime created_at "Data de disparo"
    }
```

---

## 3. Ciclo de Vida do Modelo de Machine Learning

O ciclo de vida dos modelos é gerenciado de forma determinística, impedindo que modelos subótimos alcancem o ambiente produtivo.

```mermaid
stateDiagram-v2
    [*] --> Treinamento: Trigger (Manual / Pipeline / Drift)
    
    state Treinamento {
        ValidaDataset: Validação de Qualidade (Gates)
        Otimizacao: Hyperparameter Tuning (Optuna / RandomSearch)
        CrossValidation: K-Fold CV & Generalization Gap
        AvaliacaoTeste: Métricas de Teste & Explicabilidade Shapley
        Empacotamento: Checksum SHA-256 + HMAC + Model Card
        
        ValidaDataset --> Otimizacao: Passou nos Gates
        ValidaDataset --> [*]: Falha de Qualidade (Blocked)
        Otimizacao --> CrossValidation
        CrossValidation --> AvaliacaoTeste
        AvaliacaoTeste --> Empacotamento
    }

    Empacotamento --> Candidate: Registrado no catalog.json

    state Candidate {
        [*] --> Inspecao: Revisão de Métricas
    }

    Candidate --> Staging: Promovido para homologação
    
    state Staging {
        [*] --> ReleaseGate: Avaliação Automática de Critérios
        note right of ReleaseGate
            Critérios Mínimos (Gates):
            • Acurácia >= GATE_MIN_ACCURACY
            • F1-Score >= GATE_MIN_F1
            • Latência p95 <= GATE_MAX_LATENCY_MS
        end note
    }

    ReleaseGate --> Canary: Aprovado no Gate (ou force=true)
    ReleaseGate --> Archived: Reprovado nos critérios

    state Canary {
        Roteamento: Divisão Ponderada de Tráfego
        AvaliacaoCanario: Comparação de Drift, Erros e Latência
        Roteamento --> AvaliacaoCanario
    }

    AvaliacaoCanario --> Production: Promovido (Tráfego 100%)
    AvaliacaoCanario --> Staging: Rollback Automático

    state Production {
        ServicoAtivo: Atendimento das Requisições
        MonitoramentoContinuo: Cálculo contínuo de PSI e SLO
        ServicoAtivo --> MonitoramentoContinuo
    }

    Production --> Archived: Substituído por nova versão
    Archived --> [*]
```

---

## 4. Arquitetura de Detecção de Drift e Observabilidade

O monitoramento de *drift* calcula o **Population Stability Index (PSI)** comparando a distribuição empírica das últimas 500 inferências contra a distribuição do dataset de referência (Iris baseline).

### 4.1 Formulação do PSI
O cálculo divide a feature em $B = 10$ quantis baseados na referência, com bordas infinitas nas extremidades ($-\infty, +\infty$):

$$PSI = \sum_{b=1}^{B} (A_b - E_b) \times \ln\left(\frac{A_b}{E_b}\right)$$

Onde:
- $E_b$: Proporção de observações no bin $b$ no dataset de referência (esperado).
- $A_b$: Proporção de observações no bin $b$ nas predições recentes (atual).

### 4.2 Matriz de Classificação de Drift e Ações

| Faixa de PSI | Classificação | Impacto no Health Score | Ação do Sistema |
| :--- | :--- | :--- | :--- |
| **$PSI < 0.10$** | `stable` | 0 penalidade | Operação normal contínua. |
| **$0.10 \le PSI < 0.25$** | `warning` | -15 pontos | Alerta preventivo no Prometheus; log de aviso. |
| **$PSI \ge 0.25$** | `drift` | -35 pontos | Alerta crítico; gatilho para re-treinamento automático de pipeline. |

### 4.3 Cálculo do Health Score (0 a 100)
A saúde do modelo em produção parte de **100** e sofre deduções acumuladas registradas em `health.penalties`:
- **-30 pontos:** Falha nos testes de integridade/qualidade dos dados.
- **-15 pontos:** Presença de features em estado `warning`.
- **-35 pontos:** Presença de features com `drift` crítico comprovado.
- **-15 pontos:** Violação de qualquer indicador de nível de serviço (SLO: Latência p95, Confiança média ou Acurácia com feedback).

```mermaid
flowchart LR
    subgraph Stream["Tráfego de Predições"]
        Inference["Prediction Logs<br/>(Últimas 500 predições)"]
    end

    subgraph Baseline["Baseline de Referência"]
        TrainRef["Dados de Treino / Baseline Iris"]
    end

    subgraph DriftCalc["Módulo de Análise (monitoring/drift.py)"]
        Quantiles["Bins por Quantis da Referência<br/>(Bordas abertas [-inf, +inf])"]
        PsiEngine["Cálculo de PSI por Feature<br/>sepal_length, sepal_width, petal_length, petal_width"]
        Classifier["Classificador de Severidade<br/>stable | warning | drift"]
    end

    subgraph HealthEngine["Cálculo de Saúde (monitoring/health.py)"]
        ScoreEval["Base 100 - Penalidades"]
        SloEval["SLO Monitor<br/>(Latência p95, Confiança, Feedback)"]
    end

    subgraph Exporters["Exportadores"]
        Prometheus["Gauges Prometheus<br/>iris_model_health_score<br/>iris_model_drift_psi_max"]
        AlertManager["Relatórios e Alarmes MLOps"]
    end

    Inference --> DriftCalc
    Baseline --> DriftCalc
    DriftCalc --> Quantiles --> PsiEngine --> Classifier
    Classifier --> HealthEngine
    SloEval --> HealthEngine
    HealthEngine --> ScoreEval
    ScoreEval --> Prometheus
    ScoreEval --> AlertManager
```

---

## 5. Orquestração de Pipelines e Execução em DAG

A automação de workflows é estruturada sobre um motor de Grafo Acíclico Dirigido (**DAG**) nativo (`app/orchestration/dag.py`), que valida dependências e executa tarefas em ordem topológica com tratamento de retentativas.

### 5.1 Pipeline Completo de Treinamento e Entrega (`iris-training`)

```mermaid
flowchart TD
    StartStep([Início da DAG]) --> StepQuality["1. quality_check<br/>Validação de integridade e PII"]
    
    StepQuality -->|"Aprovado"| StepTrain["2. model_training<br/>Treinamento & Busca de Hiperparâmetros"]
    StepQuality -->|"Reprovado"| BlockedEnd([Pipeline Bloqueado: status=blocked])
    
    StepTrain --> StepEval["3. model_evaluation<br/>Cross-validation, Latência p95, F1, Brier"]
    StepEval --> StepGate["4. release_gate<br/>Verificação dos limiares mínimos"]
    
    StepGate -->|"Gate Aprovado"| StepReport["5. generate_report<br/>Geração de Relatório HTML e Model Card"]
    StepGate -->|"Gate Reprovado"| FailedEnd([Falha de Critério: status=failed])
    
    StepReport --> StepAudit["6. audit_logging<br/>Gravação do evento em audit_events"]
    StepAudit --> SuccessEnd([Pipeline Concluído com Sucesso: status=completed])
```

### 5.2 Garantias de Confiabilidade e Governança
- **Reprodutibilidade:** Toda execução de pipeline armazena o `dataset_hash` e o `lineage_hash`.
- **Rastreabilidade (RBAC e Auditoria):** Cada mudança de estágio no registry requer autenticação com chave de API e papel adequado (`admin` ou `operator`), gravando o autor em `audit_events`.
- **Imutabilidade de Artefatos:** Artefatos serializados possuem lock exclusivo durante escrita e verificação de assinatura HMAC antes de cada carregamento na memória da API.

---

## 6. Arquitetura do Módulo Feature Store & Linhagem de Atributos

O módulo de Feature Store centraliza a definição, cálculo, catalogação e auditoria de variáveis preditivas. Ele garante paridade estrita entre os dados utilizados durante o treinamento em lote (*offline*) e as variáveis servidas em tempo real para inferência de baixa latência (*online*).

### 6.1 Topologia e Fluxo de Dados (Flowchart)

```mermaid
flowchart TD
    subgraph RawSources["Fontes de Dados & Ingestão"]
        SklearnData["Iris Benchmark (sklearn)"]
        CSVFiles["Custom CSV Datasets"]
        APIIngest["FastAPI / CLI Ingest Endpoints"]
        SklearnData --> SchemaCheck
        CSVFiles --> SchemaCheck
        APIIngest --> SchemaCheck
    end

    subgraph FeaturePipeline["Pipeline de Feature Store (app/features)"]
        SchemaCheck["Contratos & Validação de Esquema<br/><i>(data_catalog/contracts.py)</i>"]
        DataEnrichment["Engenharia de Features Determinística<br/><i>(features/engineering.py)</i><br/>• sepal_area = length * width<br/>• petal_area = length * width<br/>• aspect_ratios (sepal/petal)"]
        ManifestGenerator["Geração do Feature Manifest<br/><i>(tipos, estatísticas, papéis)</i>"]
        SelectionEngine["Filtragem e Seleção<br/><i>(features/selection.py)</i><br/>• VarianceThreshold<br/>• SelectKBest (ANOVA / Mutual Info)"]

        SchemaCheck --> DataEnrichment
        DataEnrichment --> ManifestGenerator
        ManifestGenerator --> SelectionEngine
    end

    subgraph StoreLayer["Persistência Híbrida do Feature Store"]
        OfflineCatalog[("Offline Store (Metadata & Datasets)<br/>SHA-256 Catalog, Artifacts Joblib")]
        OnlineStore[("Online Store (Memória / SQLite Cache)<br/>Vetores de features para inferência rápida")]
        SelectionEngine --> OfflineCatalog
        SelectionEngine --> OnlineStore
    end

    subgraph DownstreamML["Consumidores Downstream"]
        TrainRunner["Treinamento, Optuna Tuning & CV"]
        ServingEngine["FastAPI Inference Engine (/predict)"]
        DriftMonitor["PSI Drift Detector por Atributo"]

        OfflineCatalog --> TrainRunner
        OnlineStore --> ServingEngine
        ServingEngine --> DriftMonitor
    end
```

### 6.2 Estrutura de Classes e Contratos (ClassDiagram)

```mermaid
classDiagram
    class FeatureManifest {
        +int rows
        +string target
        +List~FeatureMeta~ features
        +to_dict() dict
    }
    class FeatureMeta {
        +string name
        +string dtype
        +string role
        +bool engineered
        +float min
        +float max
        +float mean
    }
    class FeatureEngineer {
        +List~string~ ENGINEERED_COLUMNS
        +enrich_iris(frame: Any) DataFrame
        +feature_manifest(frame: DataFrame, target: str) dict
    }
    class FeatureSelector {
        +variance_filter(frame: DataFrame, threshold: float) dict
        +select_k_best(frame: DataFrame, target: str, k: int, method: str) dict
    }
    class DatasetCatalog {
        +string catalog_file
        +register_dataset(name: str, frame: DataFrame) dict
        +load_dataset(name: str) DataFrame
        +get_contract(name: str) dict
    }
    class FeatureStore {
        -DatasetCatalog catalog
        -FeatureEngineer engineer
        -FeatureSelector selector
        +build_feature_view(dataset_id: str) DataFrame
        +get_online_features(entity_ids: List) DataFrame
        +publish_feature_manifest(manifest: FeatureManifest) void
    }

    FeatureEngineer ..> FeatureManifest : builds
    FeatureManifest *-- FeatureMeta : contains
    FeatureStore --> DatasetCatalog : uses
    FeatureStore --> FeatureEngineer : delegates
    FeatureStore --> FeatureSelector : selects
```

### 6.3 Orquestração e Ciclo de Vida da Feature (SequenceDiagram)

```mermaid
sequenceDiagram
    autonumber
    actor Pipeline as Orquestrador de Treinamento / API
    participant FS as Feature Store Service
    participant FE as Feature Engineer
    participant Sel as Feature Selector
    participant Cat as Dataset Catalog
    participant DB as SQLite / Metadados

    Pipeline->>FS: extract_and_transform_features(raw_data, target="species")
    activate FS
    FS->>FE: enrich_iris(raw_data)
    activate FE
    FE-->>FS: dataframe com áreas e rácios computados
    deactivate FE

    FS->>FE: feature_manifest(dataframe, target)
    activate FE
    FE-->>FS: manifest_dict (metadados estatísticos)
    deactivate FE

    FS->>Sel: select_k_best(dataframe, target, k=4, method="anova")
    activate Sel
    Sel-->>FS: selected_features, importance_scores
    deactivate Sel

    FS->>Cat: register_dataset(features, manifest)
    activate Cat
    Cat->>DB: grava hash SHA-256, schema e lineage
    DB-->>Cat: registro confirmado
    Cat-->>FS: catalog_entry_metadata
    deactivate Cat

    FS-->>Pipeline: FeaturesView(X_transformed, y, manifest, version_hash)
    deactivate FS
```

