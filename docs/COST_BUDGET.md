# Cost Budget — fraud-explanation-eval

## Per-Transaction Budget
- Maximum acceptable cost per ExplanationResult: $0.03
- Maximum acceptable cost per EvalResult: $0.05
- Combined pipeline budget per transaction: $0.08
- Budget breach action: log to cost_log.jsonl with budget_breached=True,
  increment Prometheus counter, continue (do not halt)

## Per-Agent Budget
| Agent | Model | Est. input tokens | Est. output tokens | Est. cost/call |
|-------|-------|-----------------|------------------|----------------|
| ExplanationAgent | claude-haiku-4-5 | 800 | 400 | $0.001 |
| ExplanationAgent | claude-sonnet-4-6 | 800 | 400 | $0.012 |
| EvalAgent | claude-sonnet-4-6 | 1200 | 300 | $0.015 |

Decision: Use claude-haiku for explanation (structured, bounded output),
claude-sonnet for evaluation (nuanced scoring requires stronger model).

## Phase Development Cost Log
| Phase | Date | Total API cost | Notes |
|-------|------|----------------|-------|
| 0 | - | $0.00 | Scaffold only |
