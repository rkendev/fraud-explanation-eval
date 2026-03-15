# Security Architecture — fraud-explanation-eval

## Threat Model

### Attack Surface Matrix

| # | Surface | Attack | Severity | Mitigation | Tests |
|---|---------|--------|----------|------------|-------|
| 1 | Transaction text fields (DeviceInfo, P_emaildomain, R_emaildomain) | Prompt injection via external text fields | High | `sanitize_external_text()` in `src/security/sanitizer.py` — 10 compiled regex patterns, `InjectionDetectedError` raised on match | See [Injection Tests](#1-prompt-injection-mitigation) |
| 2 | API key in log output | Secret leakage via exception messages or log records | High | `SecretRedactionFilter` in `src/utils/logging_config.py` — redacts OPENAI_API_KEY, ANTHROPIC_API_KEY, LITELLM_API_KEY from all log records | See [Secret Redaction Tests](#2-secret-redaction-mitigation) |
| 3 | FastAPI endpoints | Cost exhaustion, data harvesting via unauthenticated access | High | X-API-Key middleware in `src/api/auth.py` + rate limiting in `src/api/main.py` | See [API Auth Tests](#3-api-authentication-mitigation) |
| 4 | ExplanationResult customer mode | Probability leakage — customer sees raw fraud score | Medium | `customer_must_not_reveal_probability` model validator in `src/schemas/explanation.py` | See [Probability Leakage Tests](#4-probability-leakage-mitigation) |
| 5 | ExplanationResult feature fabrication | LLM cites features not in SHAP input (hallucination = data fabrication) | Medium | `no_hallucinated_features` field validator in `src/schemas/explanation.py` — raises ExplanationHallucinationError | See [Data Leakage Tests](#5-data-leakagefabrication-mitigation) |

### Accepted Risks
- VCR cassettes contain anonymized (but realistic) transaction feature values.
  Accepted because no real PII is present in IEEE-CIS dataset.
- SHAP values in FraudDetectionResult are exposed in analyst explanations.
  Accepted by design — analyst audience is authorized to receive them.

---

## Test References

### 1. Prompt Injection Mitigation

**Sanitizer unit tests** — `tests/unit/test_security.py`:
| Function | What it verifies |
|----------|-----------------|
| `test_clean_text_passes()` | Non-malicious text returns unchanged |
| `test_direct_injection_blocked()` | "ignore previous instructions" detected in DeviceInfo |
| `test_system_prompt_injection_blocked()` | `<system>` XML tag detected |
| `test_indirect_injection_in_email_domain_blocked()` | HTML comment injection in P_emaildomain |
| `test_empty_string_passes()` | Empty string is not a false positive |
| `test_transaction_dict_clean_passes()` | Clean transaction dict passes sanitization |
| `test_transaction_dict_injection_raises()` | "you are now a" pattern detected in dict sanitization |

**Adversarial injection tests** — `tests/adversarial/test_injection.py`:
| Function | What it verifies |
|----------|-----------------|
| `TestInjectionPatternDetection::test_ignore_instructions_detected` | "ignore previous instructions" pattern |
| `TestInjectionPatternDetection::test_disregard_rules_detected` | "disregard your previous rules" pattern |
| `TestInjectionPatternDetection::test_disregard_all_instructions_detected` | "disregard all instructions" pattern |
| `TestInjectionPatternDetection::test_you_are_now_detected` | "you are now a" pattern |
| `TestInjectionPatternDetection::test_system_prompt_detected` | "system prompt" pattern |
| `TestInjectionPatternDetection::test_act_as_role_detected` | "act as a [role]" pattern |
| `TestInjectionPatternDetection::test_act_as_if_detected` | "act as if you are" pattern |
| `TestInjectionPatternDetection::test_new_instructions_detected` | "new instructions:" pattern |
| `TestInjectionPatternDetection::test_xml_system_tag_detected` | `<system>` tag pattern |
| `TestInjectionPatternDetection::test_xml_instruction_tag_detected` | `<instruction>` tag pattern |
| `TestInjectionPatternDetection::test_inst_marker_detected` | `[INST]` marker pattern |
| `TestInjectionPatternDetection::test_hash_system_detected` | `### system` pattern |
| `TestInjectionEdgeCases::test_case_insensitive_detection` | UPPERCASE injection detected |
| `TestInjectionEdgeCases::test_mixed_case_detection` | Mixed case injection detected |
| `TestInjectionEdgeCases::test_multiline_injection_detected` | Injection on line 3 of multiline text |
| `TestInjectionEdgeCases::test_html_comment_injection_detected` | GS-004: HTML comment injection |
| `TestInjectionEdgeCases::test_injection_error_attributes` | InjectionDetectedError.source and .pattern correct |
| `TestInjectionEdgeCases::test_clean_text_passes_through` | No false positive on clean text |
| `TestInjectionEdgeCases::test_empty_string_passes` | Empty string not flagged |
| `TestTransactionFieldInjection::test_injection_in_device_info` | DeviceInfo field caught |
| `TestTransactionFieldInjection::test_injection_in_p_emaildomain` | P_emaildomain field caught |
| `TestTransactionFieldInjection::test_injection_in_r_emaildomain` | R_emaildomain field caught |
| `TestTransactionFieldInjection::test_clean_transaction_passes` | Clean transaction dict passes |

**End-to-end pipeline degradation** — `tests/adversarial/test_injection.py`:
| Function | What it verifies |
|----------|-----------------|
| `TestPipelineInjectionDegradation::test_injection_produces_degraded_result_not_crash` | Full orchestrator pipeline completes with degraded output (not crash) |
| `TestPipelineInjectionDegradation::test_injection_does_not_call_downstream_components` | Detector/explainer/evaluator never invoked after injection |
| `TestPipelineInjectionDegradation::test_email_injection_produces_degraded_result` | P_emaildomain injection also degrades gracefully |

**Integration pipeline injection test** — `tests/integration/test_pipeline.py`:
| Function | What it verifies |
|----------|-----------------|
| `TestPipelinePartialFailures::test_injection_blocked_at_sanitize` | Orchestrator routes to handle_error, detector not called |

### 2. Secret Redaction Mitigation

**Secret redaction tests** — `tests/unit/test_security.py` (via SecretRedactionFilter):
- Filter is applied by `get_logger()` factory in `src/utils/logging_config.py`
- All project loggers must use `get_logger()`, not `logging.getLogger()`
- Redaction covers message string and tuple arguments

### 3. API Authentication Mitigation

**API auth tests** — `tests/unit/test_api.py`:
- X-API-Key middleware rejects requests without valid key
- Rate limiting enforced per-client

### 4. Probability Leakage Mitigation

**Schema validator tests** — `tests/unit/test_schemas.py`:
- Customer explanation with percent format (e.g., "87%") rejected
- Customer explanation with raw float (e.g., "0.87") rejected

**Adversarial hallucination tests** — `tests/adversarial/test_hallucination.py`:
| Function | What it verifies |
|----------|-----------------|
| `test_customer_explanation_cannot_state_probability_as_percent` | "82%" in customer text rejected |
| `test_customer_explanation_cannot_state_raw_probability` | "0.82" in customer text rejected |

**Adversarial data leakage tests** — `tests/adversarial/test_data_leakage.py`:
| Function | What it verifies |
|----------|-----------------|
| `TestCustomerProbabilityLeakage::test_percent_format_rejected` | "87%" in customer text rejected |
| `TestCustomerProbabilityLeakage::test_raw_float_rejected` | "0.87" in customer text rejected |
| `TestCustomerProbabilityLeakage::test_low_probability_format_rejected` | "0.15" in customer text rejected |
| `TestCustomerProbabilityLeakage::test_borderline_probability_rejected` | "0.5" in customer text rejected |
| `TestCustomerProbabilityLeakage::test_safe_customer_text_accepted` | Clean customer text accepted |

### 5. Data Leakage/Fabrication Mitigation

**Schema validator tests** — `tests/adversarial/test_hallucination.py`:
| Function | What it verifies |
|----------|-----------------|
| `test_explanation_cannot_cite_feature_not_in_shap` | Hallucinated feature in cited_features rejected |
| `test_uncertain_explanation_must_disclose` | uncertainty_flag=True without disclosure rejected |
| `test_zero_cost_explanation_rejected` | token_cost_usd=0.0 rejected (ensures real LLM call) |

**Adversarial data leakage tests** — `tests/adversarial/test_data_leakage.py`:
| Function | What it verifies |
|----------|-----------------|
| `TestAnalystModeAllowsSHAP::test_analyst_can_state_probability` | Analyst mode CAN include probability (positive test) |
| `TestAnalystModeAllowsSHAP::test_analyst_can_cite_shap_values` | Analyst mode CAN cite SHAP values (positive test) |
| `TestFeatureFabricationLeakage::test_single_hallucinated_feature_rejected` | Single fabricated feature rejected |
| `TestFeatureFabricationLeakage::test_multiple_hallucinated_features_rejected` | Multiple fabricated features rejected |

---

## Implementation Requirements

### Sanitizer — `src/security/sanitizer.py`
- 10 compiled regex patterns (case-insensitive, multiline)
- `sanitize_external_text(text, source)` — raises `InjectionDetectedError` on match
- `sanitize_transaction_text_fields(tx_data)` — sanitizes DeviceInfo, P_emaildomain, R_emaildomain
- Called in orchestrator sanitize node before any downstream processing

### Secret Redaction — `src/utils/logging_config.py`
- `SecretRedactionFilter` redacts values of OPENAI_API_KEY, ANTHROPIC_API_KEY, LITELLM_API_KEY
- Only redacts values longer than 8 characters (avoids empty/placeholder values)
- All loggers must be created via `get_logger(name)` factory function

### Schema-Level Enforcement — `src/schemas/explanation.py`
- `no_hallucinated_features`: raises if `hallucinated_features` is non-empty
- `customer_must_not_reveal_probability`: checks percent and raw float formats
- `uncertainty_must_be_disclosed`: requires disclosure when `uncertainty_flag=True`
- `cost_must_be_real`: rejects `token_cost_usd=0.0` when explanation was generated

### Pipeline Degradation — `src/orchestrator/graph.py`
- Each node wrapped in try-except for its specific error types
- Errors populate `error` and `error_stage` in graph state
- Conditional routing sends error states to `handle_error` terminal node
- `PipelineResult` always has `completed=True`, even on partial failure
