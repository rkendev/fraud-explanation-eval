# Agent Behavior Specification — EvalAgent (LLM-as-Judge)

## Identity
- Agent: EvalAgent
- Role: Score ExplanationResult quality against rubric
- Model tier: STRONG (same tier as ExplanationAgent — evaluation requires full reasoning)
- Max execution time: 20 seconds

## Input Contract
- ExplanationResult (complete)
- FraudDetectionResult (complete — ground truth for grounding check)
- target_audience: Literal["analyst","customer"] (must match ExplanationResult)

## Output Contract
Schema: ExplanationEvalResult
- transaction_id: str
- target_audience: Literal["analyst","customer"]
- grounding_score: float (0.0–1.0) — are all claims traceable to SHAP features?
- clarity_score: float (0.0–1.0) — is explanation clear for the target audience?
- completeness_score: float (0.0–1.0) — are top fraud signals covered?
- audience_appropriateness_score: float (0.0–1.0) — correct register/terminology?
- uncertainty_handling_score: float (0.0–1.0, or null if uncertainty_flag=False)
- overall_score: float (0.0–1.0, weighted average)
- pass_threshold: float = 0.7
- passed: bool (overall_score >= pass_threshold)
- failure_reasons: list[str] (populated if passed=False)
- token_cost_usd: float

## Hallucination Constraints
- EvalAgent scores the explanation; it does not rewrite it
- EvalAgent must not invent claims about what the explanation "should have said"
  unless citing a specific SHAP feature that was omitted

## Phase Gate Requirements
- Branch coverage ≥ 85% on src/agents/eval_agent.py
- Required tests: high-score explanation passes, low-score explanation fails,
  customer explanation fails analyst rubric, uncertainty not disclosed fails,
  hallucinated feature in explanation fails grounding_score
