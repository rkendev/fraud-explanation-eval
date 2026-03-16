.PHONY: status test test-fast lint coverage train explain cost-report \
        advance-phase cassette docker-up docker-down install

# ── Dev workflow ────────────────────────────────────────────────────────────

status:
	@echo "=== Project Status ==="
	@echo "Phase:    $$(grep 'Current Phase' CLAUDE.md | head -1)"
	@echo "Tests:    $$(poetry run pytest --collect-only -q 2>/dev/null | tail -1)"
	@echo "Coverage: run 'make coverage' for details"
	@echo "CI:       check GitHub Actions"
	@echo ""

install:
	poetry install

test:
	poetry run pytest tests/ -v --cov=src --cov-report=term-missing --cov-branch

test-fast:
	poetry run pytest tests/unit/ -v --no-cov

coverage:
	poetry run pytest tests/ --cov=src --cov-report=html --cov-branch
	@echo "Open htmlcov/index.html to view branch coverage"

lint:
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/

# ── Phase management ─────────────────────────────────────────────────────────

# Usage: make advance-phase PHASE="2 — XGBoost Detector"
advance-phase:
	@if [ -z "$(PHASE)" ]; then \
		echo "Usage: make advance-phase PHASE=\"N — Name\""; exit 1; \
	fi
	@sed -i 's/- \*\*Current Phase\*\*:.*/- **Current Phase**: $(PHASE)/' CLAUDE.md
	@echo "✅ CLAUDE.md updated:"
	@grep "Current Phase" CLAUDE.md

# ── ML pipeline ──────────────────────────────────────────────────────────────

# Usage: make train SAMPLE=10000
train:
	PYTHONPATH=. poetry run python scripts/train_model.py --sample $(or $(SAMPLE),10000)

# Usage: make explain TX=TX_TEST_001
explain:
	@if [ -z "$(TX)" ]; then echo "Usage: make explain TX=<transaction_id>"; exit 1; fi
	PYTHONPATH=. poetry run python scripts/run_pipeline.py --tx $(TX)

# ── Cost monitoring ───────────────────────────────────────────────────────────

cost-report:
	poetry run python scripts/cost_report.py --log cost_log.jsonl

cost-report-phase:
	@if [ -z "$(PHASE)" ]; then echo "Usage: make cost-report-phase PHASE=phase_1"; exit 1; fi
	poetry run python scripts/cost_report.py --log cost_log.jsonl --phase $(PHASE)

# ── Testing tools ─────────────────────────────────────────────────────────────

cassette:
	poetry run pytest tests/ --record-mode=new_episodes -v

# ── Docker ───────────────────────────────────────────────────────────────────

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
