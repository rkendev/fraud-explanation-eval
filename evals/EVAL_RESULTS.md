# Evaluation Results — Golden Scenarios

**Run date**: 2026-03-16 12:50:50 UTC
**Total elapsed**: 53.7s

## Summary

| Scenario | Description | Status | Overall Score |
|----------|-------------|--------|---------------|
| GS-001 | High-value transaction from new device — high frau | PASS | 0.98 |
| GS-002 | Low-amount debit card purchase — legitimate, low f | PASS | 1.00 |
| GS-003 | Uncertain prediction — probability 0.52, confidenc | PASS | 1.00 |
| GS-004 | Injection attempt in DeviceInfo field — must be bl | SKIPPED | — |
| GS-005 | Empty SHAP features — explanation agent graceful d | SKIPPED | — |
| GS-006 | Analyst explanation must not use customer-register | PASS | 1.00 |
| GS-007 | Customer explanation must not use analyst-register | PASS | 0.92 |
| GS-008 | Budget breach — pipeline still returns result with | SKIPPED | — |
| GS-009 | EvalAgent fails low-quality explanation | FAIL | 0.24 |
| GS-010 | High-quality analyst explanation passes eval | PASS | 1.00 |

**Results**: 6 passed, 1 failed, 3 skipped, 0 errors

## Detailed Results

### GS-001: High-value transaction from new device — high fraud probability
- **Status**: PASS
- **Overall score**: 0.9840
- **Grounding**: 1.0
- **Clarity**: 0.96
- **Completeness**: 1.0
- **Audience**: 0.97
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.008109

### GS-002: Low-amount debit card purchase — legitimate, low fraud probability
- **Status**: PASS
- **Overall score**: 1.0000
- **Grounding**: 1.0
- **Clarity**: 1.0
- **Completeness**: 1.0
- **Audience**: 1.0
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.007674

### GS-003: Uncertain prediction — probability 0.52, confidence_tier=low
- **Status**: PASS
- **Overall score**: 1.0000
- **Grounding**: 1.0
- **Clarity**: 1.0
- **Completeness**: 1.0
- **Audience**: 1.0
- **Uncertainty**: 1.0
- **Passed**: True
- **Eval cost**: $0.008100

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
- **Eval cost**: $0.007647

### GS-007: Customer explanation must not use analyst-register language
- **Status**: PASS
- **Overall score**: 0.9150
- **Grounding**: 1.0
- **Clarity**: 0.95
- **Completeness**: 0.75
- **Audience**: 0.95
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.008190

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
- **Failure reasons**: ["Explanation is extremely vague and provides no analytical value — 'Transaction flagged. Could be fraud.' communicates nothing actionable to a fraud analyst.", 'Only 1 of 3 SHAP features (TransactionAmt) is cited, and even that citation is implicit at best — no SHAP value or feature value (450.0) is mentioned.', 'DeviceInfo (SHAP=0.2500, Windows 10) is entirely omitted despite being the second-highest contributor.', 'P_emaildomain (SHAP=0.1500, gmail.com) is entirely omitted despite being the third-highest contributor.', 'No fraud probability (85.0%) is referenced, nor is the high confidence tier mentioned.', 'No SHAP contribution values or magnitudes are discussed, violating analyst-audience expectations for technical precision.', "Language 'Could be fraud' is non-committal and inappropriate for an 85% high-confidence fraud prediction in an analyst-facing context.", 'Explanation fails to distinguish signal direction (positive SHAP values indicating fraud lift) for any feature.']
- **Eval cost**: $0.010551

### GS-010: High-quality analyst explanation passes eval
- **Status**: PASS
- **Overall score**: 1.0000
- **Grounding**: 1.0
- **Clarity**: 1.0
- **Completeness**: 1.0
- **Audience**: 1.0
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.007806
