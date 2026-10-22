# Iris AI Platform

Plataforma didática de ML/MLOps sobre o dataset Iris: treino, registry de modelos com verificação de
integridade, release gate, inferência, explicabilidade por instância, monitoramento e observabilidade,
expostos por API FastAPI, CLI e console web.

- [Arquitetura](arquitetura.md): módulos, fluxo de treino e de inferência, banco de dados.
- [API](api.md): grupos de rotas, autenticação e exemplos.
- [MLOps](mlops.md): métricas de treino, tuning, registry, release gate, explicabilidade e monitoramento.
- [Datasets, canário e papéis](datasets-canario-papeis.md): CSVs além do Iris, comparação de modelos com
  intervalo de confiança, deploy canário e usuários com papéis (8.1).
- [Operação](operacao.md): configuração, Docker, Prometheus/Grafana, Helm e Terraform.
- Histórico: READMEs das versões 4 a 7 e a arquitetura da 3.0, mantidos como registro.

Para começar, veja o README na raiz do repositório.
