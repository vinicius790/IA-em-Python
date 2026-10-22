# Dados

`processed/iris.csv` é uma cópia do dataset Iris do scikit-learn (150 linhas, 4 medidas em cm e a
classe). A aplicação carrega o Iris direto do scikit-learn; o CSV é usado pelo catálogo de datasets
(`POST /api/v3/datasets/register`) e pode ser recriado com:

```bash
python -m app.cli dataset
```

Em execução, esta pasta também recebe o banco SQLite (`iris.db`) e o catálogo de datasets
(`catalog.json`), ambos ignorados pelo git. Não coloque dados sensíveis no repositório.
