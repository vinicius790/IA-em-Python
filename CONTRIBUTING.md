# Guia de Contribuição — Iris AI Platform 🌸

Obrigado pelo interesse em contribuir com a **Iris AI Platform**! Siga as diretrizes abaixo para manter o rigor técnico, governança de modelos e qualidade de código do repositório.

---

## 🛠️ Regras de Engenharia e Boas Práticas

1. Use **Python 3.11 ou mais novo** e configure o ambiente com `pip install -e ".[dev]"`.
2. Antes de enviar alterações, valide a suíte completa:
   ```bash
   ruff check .
   ruff format --check .
   pytest
   ```
   *(Para rodar testes contra um PostgreSQL vazio, defina `TEST_DATABASE_URL`; veja `docs/operacao.md`)*.
3. **Toda correção de bug vem acompanhada de um teste** que falharia sem ela.
4. Rotas novas entram num router existente e herdam a autenticação; o teste `tests/test_security.py::test_every_api_route_requires_the_key` falha se uma rota ficar desprotegida.
5. Mudança de schema do banco = nova migração em `app/migrations/versions/` (há teste comparando ORM e migrações).
6. A versão única existe em `app/__init__.py`; ao mudá-la, atualize o `CHANGELOG.md` e o `appVersion` do chart Helm.
7. Integrações externas entram como extra opcional no `pyproject.toml`, com import tardio e teste dedicado.

---

## 📋 Padrão de Commits (Conventional Commits)

Seguimos estritamente o padrão [Conventional Commits](https://www.conventionalcommits.org/pt-br/v1.0.0/):

```text
<tipo>(<escopo>): <descrição no imperativo>
```

### Exemplos:
- `feat(models): adiciona suporte a classificador LightGBM`
- `fix(drift): corrige cálculo do PSI para variáveis categóricas`
- `refactor(registry): otimiza verificação de checksum SHA-256`
- `test(governance): adiciona teste para release gate com latência p95`
- `docs(readme): atualiza quick start e documentação de rotas`

---

## 🚀 Como Enviar um Pull Request (PR)

1. **Fork e Branch**:
   Crie uma branch a partir da `main`:
   ```bash
   git checkout -b feat/novo-recurso-mlops
   ```
2. **Executar Verificações**:
   Execute `ruff` e `pytest` localmente garantindo aprovação total.
3. **Commit Semântico**:
   ```bash
   git commit -m "feat(canary): adiciona suporte a alocação dinâmica de pesos"
   ```
4. **Abrir Pull Request**:
   - Forneça contexto sobre as alterações nos modelos, API ou governança.
   - Anexe os relatórios de teste gerados ou logs de validação.
