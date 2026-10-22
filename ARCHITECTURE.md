# Arquitetura do Sistema e Governança MLOps (Iris AI Platform)

A plataforma **Iris AI Platform** implementa um ecossistema completo de Engenharia de Machine Learning e MLOps, abrangendo desde a ingestão de dados e feature store até inferência em tempo real, explicabilidade (XAI), detecção estatística de data drift, auditoria automatizada e validação contínua.

---

## 1. Topologia Geral da Plataforma

```mermaid
flowchart TD
    subgraph DataIngestion["1. Ingestão de Dados & Feature Store"]
        RawData["Dataset Iris / CSVs Externos"]
        CatalogIngest["Catalog Ingestion & Schema Contracts<br/><i>(data_catalog/contracts.py)</i>"]
        QualityGate["Quality Checks & PII Redaction<br/><i>(quality.py & governance/pii.py)</i>"]
        FeatureEng["Feature Engineering & Manifest<br/><i>(features/engineering.py)</i>"]
        
        RawData --> CatalogIngest --> QualityGate --> FeatureEng
    end

    subgraph StorageLayer["2. Persistência Híbrida"]
        RelationalDb[("Banco Relacional (Alembic / SQLite / Postgres)<br/>iris_records, training_runs, prediction_logs,<br/>canary_deployments, audit_events")]
        ArtifactStore[("Model & Feature Store<br/>catalog.json, *.joblib, model_cards/*.json,<br/>lineage/*.json, audit_reports/*.md")]
        
        FeatureEng --> RelationalDb
        FeatureEng --> ArtifactStore
    end

    subgraph MLOpsCore["3. Motor de Modelagem & Avaliação Robusta"]
        TrainingEngine["ML Training & Hyperparameter Tuning<br/><i>(ml.py & optimization.py)</i>"]
        KFoldEngine["K-Fold Cross-Validation & Student-t IC<br/><i>(modeling/cross_validation.py)</i>"]
        XAIPipeline["Explainability Pipeline (XAI)<br/><i>(modeling/explainability.py)</i>"]
        AuditEngine["Model Audit & Governance Engine<br/><i>(governance/model_audit.py)</i>"]
        RegistryManager["Model Registry & Release Gates<br/><i>(registry.py & deployment/gates.py)</i>"]
        
        ArtifactStore --> TrainingEngine
        RelationalDb --> TrainingEngine
        TrainingEngine --> KFoldEngine
        TrainingEngine --> XAIPipeline
        KFoldEngine & XAIPipeline --> AuditEngine
        AuditEngine --> RegistryManager
    end

    subgraph ServingObservability["4. Inferência, Drift & Observabilidade"]
        CanaryRouter["Canary Router & Traffic Splitter<br/><i>(deployment/canary.py)</i>"]
        InferenceAPI["FastAPI Multi-Version Serving Engine<br/><i>(api_v1 ... api_v8 & cli.py)</i>"]
        DriftMonitor["Dual Statistical Drift Engine (PSI & KS-test)<br/><i>(monitoring/drift.py)</i>"]
        MetricsPrometheus["Métricas Prometheus & AlertManager<br/><i>(monitoring/health.py & sla.py)</i>"]
        
        RegistryManager --> CanaryRouter --> InferenceAPI
        InferenceAPI --> DriftMonitor --> MetricsPrometheus
        InferenceAPI --> RelationalDb
    end
```

---

## 2. Pipeline de Explicabilidade de Modelos (XAI)

O módulo de explicabilidade (`app.modeling.explainability`) combina abordagens globais e locais com base matemática rigorosa:

1. **Importância Global Nativa**: extração direta dos coeficientes ponderados em modelos lineares ($\|\beta_j\|$) ou do ganho de impureza Gini/Entropy em árvores e ensembles (`feature_importances_`).
2. **Permutação Pura (Permutation Feature Importance)**: embaralhamento aleatório com semente controlada sobre cada feature de teste/validação, computando a queda média de performance ($\Delta \text{Acc}$) e respectivo desvio padrão ($\sigma$) ao longo de $N$ iterações.
3. **Decomposição Local Aditiva (Kernel SHAP / Shapley Exato)**: para até 10 features, calcula os valores exatos de Shapley sobre todas as $2^M$ coalizões de atributos; para maiores dimensionalidades, utiliza regressão linear ponderada pelo núcleo de Shapley. Garante reconciliação aditiva estrita:
   $$\sum_{j=1}^{M} \phi_j = f(x) - \mathbb{E}[f(X)]$$
4. **Passos de Waterfall e Sumarização**: passos estruturados acumulativos da predição base até a decisão final da instância.

```mermaid
flowchart LR
    Instance["Instância de Entrada x"] --> LocalDecomp["Decomposição de Shapley<br/><i>Kernel SHAP / Exact Shapley</i>"]
    Background["Distribuição Background X"] --> BaseValue["Base Value: E[f(X)]"]
    BaseValue --> LocalDecomp
    Model["Modelo Treinado f(x)"] --> LocalDecomp
    
    LocalDecomp --> Contributions["Atribuições Locais (Phi_i)"]
    Contributions --> Reconcile["Reconciliação Aditiva<br/>sum(phi_i) = f(x) - E[f(X)]"]
    Reconcile --> Waterfall["Passos do Gráfico Waterfall<br/>& Explicação Formatada"]
```

---

## 3. Validação Cruzada K-Fold e Intervalos de Confiança

O motor `app.modeling.cross_validation` fornece validação cruzada $K$-Fold estratificada out-of-fold com estimadores clonados limpos em cada dobra:

- **Estratificação por Classe**: preserva a proporção exata de cada classe em todas as dobras de treino e validação.
- **Rastreamento de Métricas Multi-Classe**: computa Acurácia, Acurácia Balanceada, F1-Score Macro, Precisão Macro e Revocação Macro para cada fold.
- **Intervalos de Confiança (Student-t)**: calcula o intervalo bicaudal a 95% para cada métrica utilizando a distribuição t de Student baseada no desvio amostral e graus de liberdade ($K - 1$).
- **Matriz de Confusão Out-of-Fold**: sintetiza as predições de teste de todas as dobras sem vazamento de dados (*data leakage*).

```mermaid
flowchart TD
    RawDataset["Dataset de Treino (X, y)"] --> Splitter["StratifiedKFold (K=5, shuffle=True)"]
    Splitter --> Fold1["Dobra 1 (Train / Val)"]
    Splitter --> Fold2["Dobra 2 (Train / Val)"]
    Splitter --> Fold3["Dobra 3 (Train / Val)"]
    Splitter --> Fold4["Dobra 4 (Train / Val)"]
    Splitter --> Fold5["Dobra 5 (Train / Val)"]
    
    Fold1 & Fold2 & Fold3 & Fold4 & Fold5 --> Evaluator["Avaliador de Métricas (Acc, F1, Prec, Rec)"]
    Evaluator --> Aggregator["Agregador Estatístico (Média, Desvio Padrão)"]
    Aggregator --> StudentT["Intervalo de Confiança Student-t (IC 95%)"]
    Evaluator --> OOFMatrix["Matriz de Confusão Out-of-Fold"]
```

---

## 4. Detecção Estatística Bicaudal de Data Drift

O subsistema de monitoramento estatístico (`app.monitoring.drift`) implementa uma abordagem híbrida robusta para mitigação de falsos positivos e falsos negativos:

1. **PSI (Population Stability Index)**: divide a distribuição de referência em $B$ decis/percentis e compara com a distribuição atual:
   $$\text{PSI} = \sum_{b=1}^{B} \left( P_b - Q_b \right) \cdot \ln\left( \frac{P_b}{Q_b} \right)$$
   - $\text{PSI} < 0.10$: Distribuição estável.
   - $0.10 \le \text{PSI} < 0.25$: Alerta moderado (*warning*).
   - $\text{PSI} \ge 0.25$: Drift crítico detectado (*drift*).
2. **KS-Test (Kolmogorov-Smirnov bicaudal)**: teste não-paramétrico que avalia a distância suprema entre as funções de distribuição acumulada empíricas (ECDF):
   $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{cur}}(x)|$$
   Calcula o p-valor assintótico exato. Quando $p < \alpha$ (ex.: $\alpha = 0.05$), rejeita a hipótese nula de que ambas as amostras provêm da mesma distribuição contínua.
3. **Classificação Integrada**: combina a sensibilidade do KS-test com a escala de magnitude do PSI para categorizar cada atributo e o dataset geral em `stable`, `warning` ou `drift`.

```mermaid
flowchart TD
    Reference["Dados de Referência (Treino/Baseline)"] --> Preprocess["Extração de Colunas Numéricas"]
    Current["Dados Atuais de Inferência (Janela Recente)"] --> Preprocess
    
    Preprocess --> PSICalc["Cálculo do PSI por Decis"]
    Preprocess --> KSCalc["Teste de Kolmogorov-Smirnov (KS-test)"]
    
    PSICalc --> DecisionMatrix{"Matriz de Decisão Estatística"}
    KSCalc --> DecisionMatrix
    
    DecisionMatrix -->|"PSI >= 0.25 OU (PSI >= 0.10 e KS p < 0.01)"| StateDrift["Status: DRIFT (Crítico)"]
    DecisionMatrix -->|"PSI >= 0.10 OU KS p < 0.05"| StateWarning["Status: WARNING (Alerta)"]
    DecisionMatrix -->|"Caso Contrário"| StateStable["Status: STABLE (Normal)"]
```

---

## 5. Auditoria e Governança Técnica de Modelos (Model Audit Report)

O módulo `app.governance.model_audit` consolida as dimensões de confiabilidade de IA em um relatório padronizado e auditável:

```mermaid
stateDiagram-v2
    [*] --> IngestaoDados: Dados de Entrada
    IngestaoDados --> QualityGates: 1. Validação de Gates de Qualidade
    QualityGates --> KFoldValidation: 2. K-Fold Estratificado (IC 95%)
    KFoldValidation --> XAIAnalysis: 3. Explicabilidade (Global & Local)
    XAIAnalysis --> DriftMonitoring: 4. Análise de Data Drift (PSI & KS)
    DriftMonitoring --> GovernanceChecklist: 5. Checklist de Compliance
    
    state GovernanceChecklist {
        [*] --> CheckQuality: Integridade de Dados
        CheckQuality --> CheckAcc: Acurácia Média >= 85%
        CheckAcc --> CheckF1: F1-Macro Médio >= 85%
        CheckF1 --> CheckXAI: Explicabilidade Verificada
        CheckXAI --> CheckDrift: Ausência de Drift Crítico
    }
    
    GovernanceChecklist --> Approved: Todos Critérios Satisfeitos
    GovernanceChecklist --> Warning: Critérios Mínimos OK com Alertas
    GovernanceChecklist --> Rejected: Falha em Critério Crítico
    
    Approved --> ExportReport: Exportação Markdown / JSON
    Warning --> ExportReport
    Rejected --> ExportReport
    ExportReport --> [*]
```

---

## 6. Containerização Multi-Stage e Integração Contínua (CI/CD)

- **Multi-Stage Dockerfile**:
  - `builder`: ambiente com compiladores (`build-essential`) e virtualenv isolado em `/opt/venv`, instalando dependências com segurança.
  - `runtime`: imagem final ultra-enxuta baseada em `python:3.13-slim`, sem ferramentas de compilação, executando sob usuário sem privilégios (`app:app`, UID/GID 10001), com `HEALTHCHECK` configurado e diretório persistente `/app/state`.
- **GitHub Actions (`ci.yml`)**:
  - Matriz com Python 3.12 e 3.13.
  - Verificação de código com Ruff (linter e format).
  - Execução total da suíte de testes com Pytest.
  - Teste de integridade de ponta a ponta dos comandos da CLI Typer (`explain-pipeline`, `kfold`, `drift-check`, `audit-export`).
  - Smoke tests com a API em background.
  - Validação da construção da imagem Docker multi-stage via Docker Buildx.
