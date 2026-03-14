# /hallucination-check

Act as Staff QA Engineer specializing in LLM output validation.

For the ExplanationAgent, verify these invariants hold:

1. FEATURE GROUNDING
   Run: pytest tests/adversarial/test_hallucination.py -v
   Every test must pass. A failure = production hallucination risk.

2. PROBABILITY CONTAINMENT (customer mode)
   Verify: ExplanationResult with target_audience="customer" and
   explanation_text containing the fraud_probability value raises ValidationError.

3. UNCERTAINTY DISCLOSURE
   Verify: ExplanationResult with uncertainty_flag=True and
   uncertainty_disclosure=None raises ValidationError.

4. COST FIELD INTEGRITY
   Verify: ExplanationResult with token_cost_usd=0.0 raises ValidationError.

5. GOLDEN SCENARIO PASS RATE
   Run relevant golden scenarios from tests/golden/scenarios.json.
   Report: X/10 passing, list any failures with reason.

Output: PASS/FAIL per invariant + overall verdict.
