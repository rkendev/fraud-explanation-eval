# Agent Behavior Specification — ExplanationAgent

## Identity
- Agent: ExplanationAgent
- Role: Generate grounded natural-language explanation of fraud detection result
- Model tier: CHEAP (claude-haiku-4-5-20251001 — structured, bounded output)
- Max execution time: 15 seconds
- Two output modes from one schema: analyst | customer

## Input Contract
Schema: FraudDetectionResult (complete, validated — see DETECTOR_SPEC.md)
Pre-conditions:
- top_shap_features must have ≥ 1 entry (if empty, return ExplanationResult with
  explanation_generated=False and warning="insufficient_shap_data")
- fraud_probability must be in [0.0, 1.0]
- transaction_id must be non-empty

## Output Contract
Schema: ExplanationResult
- transaction_id: str
- target_audience: Literal["analyst", "customer"]
- fraud_probability: float (copied from input — must match exactly)
- is_fraud_predicted: bool (copied from input — must match exactly)
- explanation_text: str (≤300 words)
- cited_features: list[str] (feature names mentioned in explanation_text)
- uncited_features: list[str] (top_shap_features NOT mentioned — allowed)
- hallucinated_features: list[str] (features in cited_features NOT in top_shap_features)
  INVARIANT: hallucinated_features MUST always be empty list
- uncertainty_flag: bool (True if confidence_tier == "low")
- uncertainty_disclosure: Optional[str] (required if uncertainty_flag=True)
- explanation_generated: bool
- warning: Optional[str]
- token_cost_usd: float (real value from LiteLLM response — never 0.0)
- generation_latency_seconds: float

## Failure Modes
| Failure | Trigger | Response | Downstream Impact |
|---------|---------|----------|-------------------|
| Empty SHAP features | top_shap_features=[] | Return with explanation_generated=False | EvalAgent skips |
| LLM timeout | >15s | Return with explanation_generated=False, warning="llm_timeout" | Logged |
| LLM hallucinated feature | hallucinated_features non-empty | Raise ExplanationHallucinationError | Pipeline halts, alert |
| Uncertainty not disclosed | uncertainty_flag=True, uncertainty_disclosure=None | Raise UncertaintyDisclosureError | Schema validator catches |

## Hallucination Constraints
HARD RULES — enforced by Pydantic validator, not prompt:
1. cited_features must be a subset of [f.feature_name for f in top_shap_features]
   Violation = hallucinated_features non-empty = ExplanationHallucinationError
2. fraud_probability in output must equal fraud_probability in input (exact float)
3. is_fraud_predicted in output must equal is_fraud_predicted in input
4. Analyst explanation MUST NOT use language suitable only for customers
5. Customer explanation MUST NOT reveal raw fraud probability (only "flagged/not flagged")

## Prompt Strategy
Analyst system prompt includes:
- Full SHAP feature list with values
- Instruction: "You may only reference features in the provided SHAP list"
- Instruction: "State the probability as {fraud_probability:.1%}"

Customer system prompt includes:
- Top 3 SHAP features only (by |shap_value|) — no raw values, no probability
- Instruction: "Do not state any probability or score"
- Instruction: "Do not use technical terminology"

## Phase Gate Requirements
- Branch coverage ≥ 85% on src/agents/
- Required tests: analyst mode, customer mode, empty SHAP fallback,
  LLM timeout, hallucination detection (must raise), uncertainty disclosure,
  probability copy accuracy, feature citation validation
- VCR cassette required for each mode before CI integration
