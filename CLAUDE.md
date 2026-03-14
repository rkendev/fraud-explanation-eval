# CLAUDE.md — fraud-explanation-eval

## TIER 1 — ALWAYS READ (core rules, ≤150 lines)

### Project Identity
- **Name**: fraud-explanation-eval
- **Purpose**: XGBoost fraud detection + SHAP-grounded LLM explanation agent
  with full evaluation framework. The evaluation is the product.
- **Stack**: Python 3.12, XGBoost, SHAP, LangGraph, FastAPI, LiteLLM,
             Instructor+Pydantic v2, Prometheus, Docker Compose, Pytest+VCR.py
- **Dataset**: IEEE-CIS Fraud Detection (Kaggle) — named features, rich context
- **Current Phase**: 0 — Scaffold ✅
- **CI Status**: ⏳ pending first push  ← UPDATE: ✅ GREEN or ❌ RED

### Non-Negotiable Rules
1. CI must be GREEN before any feature work — if red, fix it first, nothing else
2. AGENT_SPEC.md for a component must exist before that component is implemented
3. One session = one pipeline stage — no mixing planning and implementation
4. Pydantic model defined before any agent or model code is written
5. `make status` is the first and last command every session
6. Secrets via .env only — never in conversation, never in source code, never in logs
7. ExplanationAgent receives ONLY FraudDetectionResult struct — never raw model output
8. LLM may ONLY cite features present in FraudDetectionResult.top_shap_features
9. Every agent call appends one record to cost_log.jsonl — no exceptions
10. Golden scenarios in tests/golden/scenarios.json gate every phase — not Phase 4

### End-of-Phase Protocol (non-negotiable sequence)
1. Phase complete in Claude Code → run `/agent-review` immediately (never skip)
2. Fix ALL RED items, fix AMBER items or document exceptions
3. `make test` confirms green, `make cost-report` shows no budget breach
4. Commit: `git commit -m "feat(phase-N): [name] — [X] tests, [Y]% coverage"`
5. `git push` → verify CI green
6. In terminal: `make advance-phase PHASE="N+1 — [Name]"`
7. In Claude Code: `/clear` then `/start-phase`

### Agent Contracts (immutable)
| Component          | Input                   | Output                  | Model Tier  |
|--------------------|-------------------------|-------------------------|-------------|
| DetectorModel      | FraudTransaction        | FraudDetectionResult    | No LLM      |
| ExplanationAgent   | FraudDetectionResult    | ExplanationResult       | Strong      |
| EvalAgent          | ExplanationResult+truth | ExplanationEvalResult   | Strong      |
| Orchestrator       | FraudTransaction        | manages all             | No LLM      |

### Make Targets (quick reference)
```
make status              # CI + test count + coverage + current phase
make test                # full test suite
make test-fast           # unit only
make lint                # ruff + black check
make train SAMPLE=10000  # train XGBoost on sample
make explain TX=<id>     # run full pipeline on one transaction
make cost-report         # parse cost_log.jsonl and print summary
make advance-phase PHASE="N — Name"  # update CLAUDE.md phase line
make docker-up           # start full stack
make docker-down         # stop stack
make cassette            # record VCR cassettes for explanation agent
```

## TIER 2 — READ FOR RELEVANT PHASES
@./docs/specs/DETECTOR_SPEC.md       # FraudDetectionResult contract + SHAP rules
@./docs/specs/EXPLANATION_SPEC.md    # ExplanationResult contract + hallucination rules
@./docs/specs/EVAL_SPEC.md           # EvalAgent contract + LLM-as-judge rubric
@./docs/ADR.md                       # Architecture Decision Records
@./docs/SECURITY.md                  # Threat model + mitigations
@./docs/COST_BUDGET.md               # Per-call budgets, model routing decisions
@./docs/PHASE_LOG.md                 # Completed phases + gate verdicts

## TIER 3 — READ ON DEMAND
@./docs/DATA_SOURCES.md              # IEEE-CIS schema, feature descriptions
@./.claude/skills/agent-contracts.md
@./.claude/skills/phase-context.md
@./.claude/skills/test-patterns.md
