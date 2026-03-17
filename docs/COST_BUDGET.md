# Cost Budget — fraud-explanation-eval

## Per-Transaction Budget
- Maximum acceptable cost per ExplanationResult: $0.03
- Maximum acceptable cost per EvalResult: $0.05
- Combined pipeline budget per transaction: $0.08
- Budget breach action: log to cost_log.jsonl with budget_breached=True,
  increment Prometheus counter, continue (do not halt)

## Per-Agent Budget
| Agent | Model | Est. input tokens | Est. output tokens | Est. cost/call | Actual avg cost/call |
|-------|-------|-----------------|------------------|----------------|----------------------|
| ExplanationAgent | claude-haiku-4-5-20251001 | 800 | 400 | $0.001 | $0.001770 |
| ExplanationAgent | claude-sonnet-4-6 | 800 | 400 | $0.012 | — |
| EvalAgent | claude-sonnet-4-6 | 1200 | 300 | $0.015 | $0.008306 |

Decision: Use claude-haiku for explanation (structured, bounded output),
claude-sonnet for evaluation (nuanced scoring requires stronger model).

## Phase Development Cost Log
| Phase | Date | Total API cost | Notes |
|-------|------|----------------|-------|
| 0 | - | $0.00 | Scaffold only |
| 1 | - | $0.00 | Data pipeline, no LLM calls |
| 2 | - | $0.00 | XGBoost + SHAP, no LLM calls |
| 3 | - | $0.11 | ExplanationAgent first wired — phase_3 share = $0.11149 |
| 4 | - | $0.43 | EvalAgent wired — 52 calls, phase_4 share = $0.43191 |
| 5 | - | $0.00 | Orchestrator, no new LLM calls |
| 6 | - | $0.00 | FastAPI wrapper, no new LLM calls |
| 7 | - | $0.00 | Hardening, no new LLM calls |
| **Total** | | **$0.54** | 106 calls across 26 transactions |
