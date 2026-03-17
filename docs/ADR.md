# Architecture Decision Records

## ADR-001: XGBoost over neural network for fraud detection
Date: Phase 0
Decision: Use XGBoost, not a neural network (LSTM, Transformer).
Rationale: XGBoost produces per-prediction SHAP values natively. Neural networks
require post-hoc approximation (LIME, Integrated Gradients) which introduces
a second source of error between model decision and explanation. The explanation
quality is the product — the detector is the substrate. SHAP accuracy is paramount.
Rejected: LightGBM (similar but less SHAP ecosystem support), neural network.

## ADR-002: SHAP top-5 features as the only LLM input
Date: Phase 0
Decision: ExplanationAgent receives only top 5 SHAP features, not all 400+ IEEE-CIS fields.
Rationale: Prevents the LLM from citing non-contributory features as if they were
relevant. The hallucination detection mechanism (hallucinated_features validator) only
works if the permitted citation list is bounded. An unbounded input = unbounded
hallucination surface.
Rejected: Passing all features (too much context, hallucination risk),
passing top 10 (reasonable alternative — revisit in Phase 3).

## ADR-003: Two audience modes from one ExplanationResult schema
Date: Phase 0
Decision: target_audience: Literal["analyst","customer"] drives prompt selection.
One schema, two prompts, validated by different field rules per audience.
Rationale: Single schema = single test surface = simpler golden scenario structure.
Customer mode enforces probability suppression at the schema level (not just prompt),
preventing accidental leakage of fraud_probability to customers.

## ADR-004: LangGraph for orchestration
Date: Phase 0
Decision: Use LangGraph state machine to connect DetectorModel → ExplanationAgent → EvalAgent.
Rationale: Auditable state transitions, built-in node-level retry, compatible with
existing portfolio pattern. Orchestrator has no LLM call — pure routing.
Rejected: Raw async Python (less auditable), CrewAI (heavier dependency, less control).

## ADR-005: IEEE-CIS over Kaggle credit card dataset
Date: Phase 0
Decision: IEEE-CIS for rich named features; Kaggle CC dataset rejected.
Rationale: Kaggle CC dataset has 28 PCA-anonymized features (V1-V28). The LLM
cannot generate a grounded explanation citing "V3 was elevated" — that's meaningless
to any audience. IEEE-CIS has DeviceType, P_emaildomain, ProductCD, card4 — features
a human fraud analyst would actually reference.

## ADR-006: cost_log.jsonl as mandatory cost audit trail
Date: Phase 0
Decision: Every LLM call appends one structured record to cost_log.jsonl.
Rationale: token_cost_usd field existed in prior project but was never populated.
This ADR makes cost logging non-optional at the architecture level, not the
implementation level. Grafana dashboard and make cost-report both depend on it.

## ADR-007: Model version string is claude-haiku-4-5-20251001
Date: Phase 3
Decision: Use the full versioned model string in LiteLLM calls.
Rationale: LiteLLM requires the full versioned string for correct routing.
The COST_PER_TOKEN table was updated in the audit fix (P2) to include
the versioned key. COST_BUDGET.md previously referenced the short form —
now corrected to match the actual model string used.

## ADR-008: cost_must_be_real validator exempts explanation_generated=False
Date: Phase 3
Decision: token_cost_usd=0.0 is valid when explanation_generated=False
(degraded result — no LLM call was made).
Rationale: A timeout or injection-blocked result produces no tokens.
Rejecting 0.0 in that case would prevent valid degraded results from
being constructed. The validator checks explanation_generated first.
Rejected: Always requiring > 0.0 (would break degraded mode).

## ADR-009: pyarrow added as runtime dependency
Date: Phase 1
Decision: Add pyarrow to pyproject.toml main dependencies.
Rationale: pandas .to_parquet() and pd.read_parquet() require pyarrow.
Discovered when test_saves_parquet failed in CI on a clean environment.
Rejected: fastparquet (less widely supported).

## ADR-010: ruff N803/N806 suppressed for Pydantic validators
Date: Phase 0 (applied throughout)
Decision: Suppress ruff rules N803 (argument name should be lowercase)
and N806 (variable in function should be lowercase) in pyproject.toml.
Rationale: Pydantic v2 @field_validator methods use cls as the first
argument and Pydantic-specific uppercase patterns. ruff's naming rules
conflict with Pydantic conventions. Suppression is scoped to this pattern.

## ADR-011: TX_001 triple zero-cost records are a test-environment artifact
Date: Phase 7 (discovered in audit)
Decision: Accept the three zero-cost TX_001 records in cost_log.jsonl
as a known test-environment artifact. Not fixing in this iteration.
Rationale: Three identical zero-cost ExplanationAgent records for TX_001
appear twice (microsecond-apart timestamps). This is a concurrent
test execution pattern — record_agent_call() is called before the LLM
response returns tokens in certain test harness configurations.
This does not affect production pipeline runs. The pre-commit hook
correctly flags zero-cost records; these were committed from a test
environment where the hook was not active.
