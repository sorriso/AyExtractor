# Makefile — v3
# === ayExtractor Makefile ===

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
SRC_DIR = src
TEST_DIR = tests
COV_DIR = coverage

.PHONY: help install test test-unit test-integration test-gpu test-coverage lint format typecheck clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install project in editable mode with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]" --break-system-packages

test: test-unit test-integration  ## Run all tests (unit + integration)

test-unit:  ## Run unit tests only (no external deps required)
	$(PYTEST) $(TEST_DIR)/unit/ -v --tb=short

test-integration:  ## Run integration tests (requires LLM API keys + optional DBs)
	$(PYTEST) $(TEST_DIR)/integration/ -v --tb=short

test-gpu:  ## Run GPU-accelerated tests (requires NVIDIA GPU + RAPIDS)
	$(PYTEST) $(TEST_DIR)/ -v --tb=short -m gpu

test-coverage:  ## Run unit tests with coverage report (terminal + HTML + XML)
	@mkdir -p $(COV_DIR)
	$(PYTEST) $(TEST_DIR)/unit/ \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html:$(COV_DIR)/html \
		--cov-report=xml:$(COV_DIR)/coverage.xml \
		--cov-report=json:$(COV_DIR)/coverage.json \
		-v --tb=short
	@echo ""
	@echo "Coverage reports generated in $(COV_DIR)/"
	@echo "  HTML  → $(COV_DIR)/html/index.html"
	@echo "  XML   → $(COV_DIR)/coverage.xml"
	@echo "  JSON  → $(COV_DIR)/coverage.json"

lint:  ## Run linters (ruff)
	$(PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR)

format:  ## Auto-format code (ruff)
	$(PYTHON) -m ruff format $(SRC_DIR) $(TEST_DIR)

typecheck:  ## Run type checker (mypy)
	$(PYTHON) -m mypy $(SRC_DIR)

clean:  ## Remove build artifacts, caches, coverage
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov/ .coverage $(COV_DIR)/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true