# Skill: Test Patterns for fraud-explanation-eval

## VCR Cassette Pattern (for ExplanationAgent)
Use pytest-recording (VCR.py wrapper). Cassettes in tests/cassettes/explanation/.
Naming: {agent}_{audience}_{scenario}_{YYYY-MM}.yaml
Example: explanation_analyst_high_fraud_2026-03.yaml

## Hallucination Test Pattern
1. Construct a FraudDetectionResult with known top_shap_features
2. Attempt to create ExplanationResult with hallucinated_features non-empty
3. Assert ValidationError is raised with "ExplanationHallucinationError" message
Do NOT test with a live LLM call — test the schema validator directly.

## Cost Tracking Test Pattern
1. Run agent call with mocked LiteLLM response (known token counts)
2. Assert cost_log.jsonl was appended with correct record
3. Assert returned token_cost_usd matches calculated value
4. Assert token_cost_usd on schema is > 0.0

## Golden Scenario Test Pattern
1. Load tests/golden/scenarios.json
2. For each scenario: construct inputs per scenario spec
3. Assert output matches expected fields (probability range, flags, features)
4. Assert no disqualifying hallucination conditions are met
5. These are integration tests — they require live model (mocked in CI)

## Security Test Pattern
Any test involving DeviceInfo or email domain fields:
1. Test clean input passes sanitize_external_text()
2. Test injection pattern raises InjectionDetectedError
3. Test that ExplanationResult is NOT generated after injection detected
