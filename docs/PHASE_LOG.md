# Phase Log

## Phase 0 — Scaffold
Status: ✅ COMPLETE
Gate verdict: GREEN
Deliverables:
- CLAUDE.md (tiered, Tier 1 ≤150 lines)
- All 4 Pydantic schemas (FraudTransaction, FraudDetectionResult, ExplanationResult, ExplanationEvalResult)
- Security module (sanitizer.py, logging_config.py)
- Cost tracker (cost_tracker.py, scripts/cost_report.py)
- 10 golden scenarios (tests/golden/scenarios.json)
- 20+ schema and security tests
- Populated spec docs (DETECTOR_SPEC, EXPLANATION_SPEC, EVAL_SPEC, ADR, SECURITY, COST_BUDGET)
- CI workflow with zero-cost validation and secret check
- make advance-phase target

## Phase 1 — Data Pipeline
Status: ✅ COMPLETE
Gate verdict: GREEN
Tests at gate: 77
Coverage: 97%
Deliverables:
- src/data/loader.py (IEEE-CIS ingestion + feature engineering)
- src/data/preprocessor.py (train/test split, class balancing with SMOTE)
- tests/unit/test_data_pipeline.py (45 tests)
- data/processed/ (train.parquet, test.parquet)
- pyarrow added as runtime dependency (see ADR-009)

## Phase 2 — XGBoost Detector
Status: ✅ COMPLETE
Gate verdict: GREEN
Tests at gate: 131
Coverage: 95%
Deliverables:
- src/models/detector.py (XGBoost training + inference)
- src/models/shap_extractor.py (top-5 SHAP features)
- models/artifacts/model.json + version.txt
- tests/unit/test_detector.py (54 tests)
- Confidence tier boundary tests at exact thresholds (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
- All failure modes tested (ModelNotLoadedError, TransactionValidationError, InferenceTimeoutError, SHAP failure)

## Phase 3 — Explanation Agent
Status: ✅ COMPLETE
Gate verdict: AMBER
Tests at gate: 178
Coverage: 95%
Deliverables:
- src/agents/explanation_agent.py (both audience modes)
- tests/unit/test_explanation_agent.py (47 tests)
- tests/cassettes/explanation/ (VCR cassettes for analyst + customer modes)
- Hallucination detection tests pass
- Customer probability leakage test passes
- cost_log.jsonl phase_3 total: $0.11149
AMBER exceptions:
- logging.getLogger() used instead of get_logger() in some modules; documented, not a blocker

## Phase 4 — Evaluation Framework
Status: ✅ COMPLETE
Gate verdict: GREEN
Tests at gate: 222
Coverage: 96%
Deliverables:
- src/agents/eval_agent.py (LLM-as-judge)
- tests/unit/test_eval_agent.py (44 tests)
- evals/run_golden_scenarios.py (runs all 10 golden scenarios)
- evals/EVAL_RESULTS.md (output report)
- 6/7 golden scenarios pass (GS-009 fails by design — proves evaluator rejects bad explanations)
- cost_log.jsonl phase_4 total: $0.43191

## Phase 5 — Orchestrator + LangGraph
Status: ✅ COMPLETE
Gate verdict: AMBER
Tests at gate: 242
Coverage: 96%
Deliverables:
- src/orchestrator/graph.py (LangGraph state machine — 5 nodes, error routing)
- src/orchestrator/state.py (GraphState + PipelineResult)
- tests/integration/test_pipeline.py (20 tests)
- Full pipeline: TX → FraudDetectionResult → ExplanationResult → EvalResult
- Partial failure (injection blocked) produces correct degraded output
AMBER exceptions:
- Golden scenarios tested via mock patterns, not from scenarios.json directly; full integration deferred to Phase 6 when make explain is available

## Phase 6 — FastAPI + Observability
Status: ✅ COMPLETE
Gate verdict: GREEN (4/4 roles GREEN)
Tests at gate: 260
Coverage: 96%
Deliverables:
- src/api/main.py (SSE streaming, API key auth, rate limiting)
- src/api/auth.py (X-API-Key middleware)
- src/api/schemas.py (API request/response models)
- prometheus/ config
- grafana/dashboards/cost_dashboard.json (6 panels)
- Docker Compose with all services
- API rejects unauthenticated requests (401)

## Phase 7 — Hardening
Status: ✅ COMPLETE
Gate verdict: GREEN
Tests at gate: 295
Coverage: 97%
Deliverables:
- tests/adversarial/test_injection.py (26 tests — all 10 injection patterns + pipeline degradation)
- tests/adversarial/test_data_leakage.py (9 tests — probability leakage + feature fabrication)
- docs/SECURITY.md fully populated with specific test file and function references
- README.md with full pipeline architecture diagram
- ExplanationAgent timeout path fixed to call record_agent_call() per Rule 9
- All adversarial tests pass, security checklist complete
