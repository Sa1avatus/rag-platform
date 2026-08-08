.PHONY: up down test lint migrate seed
up:
	docker compose up -d --build
down:
	docker compose down
test:
	pytest
lint:
	ruff check . && mypy src
migrate:
	alembic upgrade head
seed:
	python -m rag_platform.seed
