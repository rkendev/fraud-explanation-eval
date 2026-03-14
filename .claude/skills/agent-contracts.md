# Skill: Agent Output Contracts (production-locked)

## FraudTransaction (input)
Key fields: TransactionID (str, non-empty), TransactionAmt (float, >0),
ProductCD (W/H/C/S/R), card4, card6, addr1, P_emaildomain, R_emaildomain,
DeviceType (desktop|mobile), DeviceInfo
Validation: DeviceInfo/emaildomain fields are sanitized via sanitize_external_text()
BEFORE entering LLM context. Any injection attempt raises InjectionDetectedError.

## FraudDetectionResult (XGBoost output)
Key fields: transaction_id, fraud_probability (0-1), is_fraud_predicted,
top_shap_features (list[SHAPFeature], max 5),
confidence_tier (high|medium|low), model_version, inference_latency_ms
Invariant: confidence_tier MUST be consistent with fraud_probability ranges.
Invariant: is_fraud_predicted MUST be consistent with fraud_probability >= 0.5.

## ExplanationResult (LLM output)
Key fields: transaction_id, target_audience (analyst|customer),
fraud_probability (copied from detection — must match exactly),
is_fraud_predicted (copied — must match), explanation_text (≤300 words),
cited_features, uncited_features, hallucinated_features (MUST be empty),
uncertainty_flag, uncertainty_disclosure (required if flag=True),
token_cost_usd (MUST be > 0.0 — real LiteLLM value), generation_latency_seconds
HARD INVARIANT: hallucinated_features must ALWAYS be empty.
HARD INVARIANT: customer mode must NEVER contain fraud_probability value.
HARD INVARIANT: token_cost_usd must NEVER be 0.0.

## ExplanationEvalResult (LLM-as-judge output)
Key fields: grounding_score, clarity_score, completeness_score,
audience_appropriateness_score, uncertainty_handling_score (null if N/A),
overall_score, passed (must be consistent with overall_score >= 0.70),
failure_reasons, token_cost_usd

## Cost Tracking (mandatory)
Every LLM call must call record_agent_call() from src/utils/cost_tracker.py.
This appends to cost_log.jsonl and updates Prometheus metrics.
The returned cost_usd value must be set on the schema's token_cost_usd field.
