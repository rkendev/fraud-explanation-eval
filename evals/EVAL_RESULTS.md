# Evaluation Results — Golden Scenarios

**Run date**: 2026-03-15 15:27:25 UTC
**Total elapsed**: 53.9s

## Summary

| Scenario | Description | Status | Overall Score |
|----------|-------------|--------|---------------|
| GS-001 | High-value transaction from new device — high frau | PASS | 0.98 |
| GS-002 | Low-amount debit card purchase — legitimate, low f | PASS | 1.00 |
| GS-003 | Uncertain prediction — probability 0.52, confidenc | PASS | 1.00 |
| GS-004 | Injection attempt in DeviceInfo field — must be bl | SKIPPED | — |
| GS-005 | Empty SHAP features — explanation agent graceful d | SKIPPED | — |
| GS-006 | Analyst explanation must not use customer-register | PASS | 1.00 |
| GS-007 | Customer explanation must not use analyst-register | PASS | 0.93 |
| GS-008 | Budget breach — pipeline still returns result with | SKIPPED | — |
| GS-009 | EvalAgent fails low-quality explanation | FAIL | 0.24 |
| GS-010 | High-quality analyst explanation passes eval | PASS | 0.99 |

**Results**: 6 passed, 1 failed, 3 skipped, 0 errors

## Detailed Results

### GS-001: High-value transaction from new device — high fraud probability
- **Status**: PASS
- **Overall score**: 0.9775
- **Grounding**: 1.0
- **Clarity**: 0.95
- **Completeness**: 1.0
- **Audience**: 0.95
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.008121

### GS-002: Low-amount debit card purchase — legitimate, low fraud probability
- **Status**: PASS
- **Overall score**: 1.0000
- **Grounding**: 1.0
- **Clarity**: 1.0
- **Completeness**: 1.0
- **Audience**: 1.0
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.007470

### GS-003: Uncertain prediction — probability 0.52, confidence_tier=low
- **Status**: PASS
- **Overall score**: 1.0000
- **Grounding**: 1.0
- **Clarity**: 1.0
- **Completeness**: 1.0
- **Audience**: 1.0
- **Uncertainty**: 1.0
- **Passed**: True
- **Eval cost**: $0.008241

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
- **Eval cost**: $0.008106

### GS-007: Customer explanation must not use analyst-register language
- **Status**: PASS
- **Overall score**: 0.9250
- **Grounding**: 1.0
- **Clarity**: 0.95
- **Completeness**: 0.75
- **Audience**: 1.0
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.009036

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
- **Failure reasons**: ["Explanation is extremely vague and provides almost no analytical value — 'Transaction flagged. Could be fraud.' offers no actionable insight for a fraud analyst.", 'Only one of three SHAP features (TransactionAmt) is cited, and even then it is not actually discussed or quantified in the explanation body.', "Critical SHAP features 'DeviceInfo' (shap_value=0.2500) and 'P_emaildomain' (shap_value=0.1500) are entirely omitted with no mention or analysis.", 'The fraud probability (85.0%) is not referenced anywhere in the explanation, which is a core requirement for analyst-facing outputs.', 'SHAP contribution values are not referenced at all, making the explanation useless for technical fraud analysis.', "Language is far too casual and non-committal ('Could be fraud') for the target analyst audience — it fails to convey the high-confidence (85%) fraud prediction with appropriate precision.", 'No structured reasoning or logical flow is present; the explanation lacks the register expected by a fraud analyst.']
- **Eval cost**: $0.010446

### GS-010: High-quality analyst explanation passes eval
- **Status**: PASS
- **Overall score**: 0.9890
- **Grounding**: 1.0
- **Clarity**: 0.98
- **Completeness**: 1.0
- **Audience**: 0.97
- **Uncertainty**: N/A
- **Passed**: True
- **Eval cost**: $0.008082
