# Security Architecture — fraud-explanation-eval

## Threat Model

### Attack Surface Matrix
| # | Surface | Attack | Severity | Mitigation | Test |
|---|---------|--------|----------|------------|------|
| 1 | Transaction description fields | Prompt injection via DeviceInfo or P_emaildomain | High | sanitize_external_text() before LLM context | tests/adversarial/test_injection.py |
| 2 | API key in LLM context | Secret leakage via exception messages | High | SecretRedactionFilter on all loggers | tests/unit/test_security.py |
| 3 | FastAPI endpoint unauthenticated | Cost exhaustion, data harvesting | High | X-API-Key middleware + rate limiting | tests/integration/test_api_auth.py |
| 4 | ExplanationResult customer mode | Probability leakage to customer audience | Medium | Schema validator prohibits probability field for customer | tests/unit/test_schemas.py |
| 5 | SHAP feature values in explanation | Feature value exfiltration via analyst explanation | Medium | Analyst mode only served to authenticated analysts | tests/adversarial/test_data_leakage.py |

### Accepted Risks
- VCR cassettes contain anonymized (but realistic) transaction feature values.
  Accepted because no real PII is present in IEEE-CIS dataset.
- SHAP values in FraudDetectionResult are exposed in analyst explanations.
  Accepted by design — analyst audience is authorized to receive them.

## Implementation Requirements
See src/security/sanitizer.py for injection detection.
See src/utils/logging_config.py for SecretRedactionFilter.
All loggers must be initialized via get_logger() in logging_config.py.
