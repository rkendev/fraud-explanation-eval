# Evaluation Results — Golden Scenarios

**Run date**: 2026-03-16 10:04:17 UTC
**Total elapsed**: 56.0s

## Summary

| Scenario | Description | Status | Overall Score |
|----------|-------------|--------|---------------|
| GS-001 | High-value transaction from new device — high frau | PASS | 1.00 |
| GS-002 | Low-amount debit card purchase — legitimate, low f | PASS | 1.00 |
| GS-003 | Uncertain prediction — probability 0.52, confidenc | PASS | 0.99 |
| GS-004 | Injection attempt in DeviceInfo field — must be bl | SKIPPED | — |
| GS-005 | Empty SHAP features — explanation agent graceful d | SKIPPED | — |
| GS-006 | Analyst explanation must not use customer-register | PASS | 1.00 |
| GS-007 | Customer explanation must not use analyst-register | PASS | 0.92 |
| GS-008 | Budget breach — pipeline still returns result with | SKIPPED | — |
| GS-009 | EvalAgent fails low-quality explanation | FAIL | 0.24 |
| GS-010 | High-quality analyst explanation passes eval | PASS | 0.99 |

**Results**: 6 passed, 1 failed, 3 skipped, 0 errors

## Detailed Results

### GS-001: High-value transaction from new device — high fraud probability
- **Status**: PASS
- **Overall score**: 1.0000
- **Grounding**: 1.0
- **Clarity**: 1.0
- **Completeness**: 1.0
- **Audience**: 1.0
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.008037

### GS-002: Low-amount debit card purchase — legitimate, low fraud probability
- **Status**: PASS
- **Overall score**: 1.0000
- **Grounding**: 1.0
- **Clarity**: 1.0
- **Completeness**: 1.0
- **Audience**: 1.0
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.007776

### GS-003: Uncertain prediction — probability 0.52, confidence_tier=low
- **Status**: PASS
- **Overall score**: 0.9925
- **Grounding**: 1.0
- **Clarity**: 1.0
- **Completeness**: 1.0
- **Audience**: 0.95
- **Uncertainty**: 1.0
- **Passed**: True
- **Eval cost**: $0.008196

### GS-004: Injection attempt in DeviceInfo field — must be blocked
- **Status**: SKIPPED
- **Reason**: Scenario tests injection/empty-SHAP/budget — not EvalAgent scope

### GS-005: Empty SHAP features — explanation agent graceful degradation
- **Status**: SKIPPED
- **Reason**: Scenario tests injection/empty-SHAP/budget — not EvalAgent scope

### GS-006: Analyst explanation must not use customer-register language
- **Status**: PASS
- **Overall score**: 1.0000
- **Grounding**: 1.0
- **Clarity**: 1.0
- **Completeness**: 1.0
- **Audience**: 1.0
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.008058

### GS-007: Customer explanation must not use analyst-register language
- **Status**: PASS
- **Overall score**: 0.9150
- **Grounding**: 1.0
- **Clarity**: 0.95
- **Completeness**: 0.75
- **Audience**: 0.95
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.009933

### GS-008: Budget breach — pipeline still returns result with flag
- **Status**: SKIPPED
- **Reason**: Scenario tests injection/empty-SHAP/budget — not EvalAgent scope

### GS-009: EvalAgent fails low-quality explanation
- **Status**: FAIL
- **Overall score**: 0.2450
- **Grounding**: 0.5
- **Clarity**: 0.2
- **Completeness**: 0.1
- **Audience**: 0.1
- **Uncertainty**: N/A
- **Passed**: False
- **Failure reasons**: ["Extremely vague explanation ('Transaction flagged. Could be fraud.') provides no actionable analytical value.", 'Only one of three SHAP features (TransactionAmt) is cited, and even then it is not actually discussed or referenced with its value (450.0) or SHAP contribution (0.35).', "DeviceInfo (SHAP=0.25, value='Windows 10') is completely omitted despite being the second most important signal.", "P_emaildomain (SHAP=0.15, value='gmail.com') is completely omitted despite being the third most important signal.", 'The fraud probability (85.0%) and confidence tier (high) are not mentioned anywhere in the explanation.', 'No SHAP values or contribution magnitudes are cited, which is required for a fraud analyst audience.', "The language ('Could be fraud') is inappropriately hedging given a high-confidence 85% fraud prediction and does not reflect the model's output accurately.", 'The explanation uses casual, non-technical language entirely unsuitable for a fraud analyst — no reference to model scores, SHAP contributions, or feature-level evidence.']
- **Eval cost**: $0.011361

### GS-010: High-quality analyst explanation passes eval
- **Status**: PASS
- **Overall score**: 0.9890
- **Grounding**: 1.0
- **Clarity**: 0.98
- **Completeness**: 1.0
- **Audience**: 0.97
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.008172
