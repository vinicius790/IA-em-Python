# tests/

Suíte de testes automatizados (pytest) da plataforma.

- `test_api*.py`: endpoints FastAPI e contrato OpenAPI.
- Demais `test_*.py`: serviços de ML, segurança, CLI, banco e pipelines.
- `conftest.py`: fixtures compartilhadas.

Executar:

```bash
pip install -e ".[dev]"
pytest
```

O CI exige cobertura mínima de 90%.
