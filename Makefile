.PHONY: install dev lint format test clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	mypy providers services workflows

format:
	ruff format .
	ruff check --fix .

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

run:
	uvicorn api_service:app --reload --host 0.0.0.0 --port 8000
