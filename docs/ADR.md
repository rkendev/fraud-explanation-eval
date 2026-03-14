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
