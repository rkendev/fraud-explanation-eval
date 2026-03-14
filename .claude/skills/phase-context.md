# Skill: Build Phase Context

## Phase 0 — Scaffold (COMPLETE)
Deliverables: CLAUDE.md, all Pydantic schemas, security module, cost tracker,
  golden scenarios (10), test files (unit + adversarial), CI, Makefile,
  populated spec docs (DETECTOR_SPEC, EXPLANATION_SPEC, EVAL_SPEC, ADR, SECURITY, COST_BUDGET)
Gate: CI green, 20+ tests passing, make status clean

## Phase 1 — Data Pipeline
Deliverables: src/data/loader.py (IEEE-CIS ingestion + feature engineering),
  src/data/preprocessor.py (train/test split, class balancing with SMOTE),
  tests/unit/test_data_pipeline.py (20+ tests),
  data/processed/ (train.parquet, test.parquet)
Gate: 20+ tests, data schema validated, no PII in processed files

## Phase 2 — XGBoost Detector
Deliverables: src/models/detector.py (XGBoost training + inference),
  src/models/shap_extractor.py (top-5 SHAP features),
  models/artifacts/model.json + version.txt,
  tests/unit/test_detector.py (25+ tests),
  make train produces FraudDetectionResult for sample transaction
Gate: 25+ tests, FraudDetectionResult validated against contract,
  all failure modes tested, confidence tier boundaries tested

## Phase 3 — Explanation Agent
Deliverables: src/agents/explanation_agent.py (both audience modes),
  tests/unit/test_explanation_agent.py (25+ tests),
  tests/cassettes/explanation/ (VCR cassettes for analyst + customer modes)
Gate: 25+ tests, hallucination tests pass, VCR cassettes for both modes,
  zero-cost validation enforced, customer probability leakage test passes

## Phase 4 — Evaluation Framework
Deliverables: src/agents/eval_agent.py (LLM-as-judge),
  tests/unit/test_eval_agent.py (20+ tests),
  evals/run_golden_scenarios.py (runs all 10 golden scenarios),
  evals/EVAL_RESULTS.md (output report)
Gate: 10/10 golden scenarios produce ExplanationEvalResult,
  LLM-as-judge scores match human expectations on 3+ hand-verified cases

## Phase 5 — Orchestrator + LangGraph
Deliverables: src/orchestrator/graph.py (LangGraph state machine),
  src/orchestrator/state.py (GraphState),
  tests/integration/test_pipeline.py (15+ tests)
Gate: Full pipeline TX → FraudDetectionResult → ExplanationResult → EvalResult,
  partial failure (injection blocked) produces correct degraded output

## Phase 6 — FastAPI + Observability
Deliverables: src/api/main.py (SSE streaming, API key auth, rate limiting),
  prometheus/ config, grafana/dashboards/cost_dashboard.json,
  Docker Compose with all services
Gate: make explain TX=TX_TEST_001 produces full output,
  Grafana cost dashboard shows 6 panels, API rejects unauthenticated requests

## Phase 7 — Hardening
Deliverables: tests/adversarial/ (extended injection tests, data leakage tests),
  docs/SECURITY.md fully populated with all test references,
  README.md with architecture diagram
Gate: 5+ adversarial tests pass, security checklist complete,
  make cost-report shows all phase costs
