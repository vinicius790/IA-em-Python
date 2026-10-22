.PHONY: install test lint format run seed train pipeline quality smoke docs migrate docker-up docker-down

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

run:
	uvicorn app.main:app --reload

seed:
	python -m app.cli seed

# --tune usa Optuna se o extra `tuning` estiver instalado; senão, RandomizedSearchCV.
train:
	python -m app.cli train --model xgboost --tune

pipeline:
	python -m app.cli pipeline --model xgboost

quality:
	python -m app.cli quality

# Exige a API em execução (make run) em http://127.0.0.1:8000.
smoke:
	python scripts/smoke_test.py

docs:
	pip install -e ".[docs]"
	mkdocs build --strict

migrate:
	alembic upgrade head

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
