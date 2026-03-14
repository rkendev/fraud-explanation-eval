# fraud-explanation-eval

> XGBoost fraud detection + SHAP-grounded LLM explanation evaluation framework.
> **The evaluation is the product. The detector is the substrate.**

## Architecture

```
FraudTransaction (IEEE-CIS fields)
    │
    ▼ sanitize_external_text()   ← injection prevention
DetectorModel (XGBoost + SHAP)
    │
    ▼ FraudDetectionResult
    │  ├── fraud_probability
    │  ├── is_fraud_predicted
    │  ├── confidence_tier
    │  └── top_shap_features (top 5 by |SHAP value|)
    │
ExplanationAgent (LiteLLM, cheap model)
    │  ├── target_audience: "analyst" → cites SHAP values, states probability
    │  └── target_audience: "customer" → no probability, plain language
    │
    ▼ ExplanationResult
    │  ├── hallucinated_features (INVARIANT: always empty)
    │  ├── token_cost_usd (INVARIANT: always > 0.0)
    │  └── uncertainty_disclosure (required if confidence_tier == "low")
    │
EvalAgent (LiteLLM-as-judge, strong model)
    │
    ▼ ExplanationEvalResult
       ├── grounding_score
       ├── clarity_score
       ├── audience_appropriateness_score
       └── passed (threshold: 0.70)
```

## Quick Start

```bash
cp .env.example .env   # fill in API keys
poetry install
make test              # should show 20+ tests passing
make status
```

## Build Phases

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Scaffold, schemas, specs, golden scenarios | ✅ Complete |
| 1 | Data pipeline (IEEE-CIS ingestion + feature engineering) | ⏳ |
| 2 | XGBoost detector + SHAP extractor | ⏳ |
| 3 | Explanation agent (analyst + customer modes) | ⏳ |
| 4 | Evaluation framework (LLM-as-judge + golden scenarios) | ⏳ |
| 5 | Orchestrator + LangGraph state machine | ⏳ |
| 6 | FastAPI + cost dashboard + observability | ⏳ |
| 7 | Hardening (adversarial tests, security checklist) | ⏳ |

## What Makes This Different

- **Hallucination is structurally prevented**: The `hallucinated_features` Pydantic
  validator raises `ExplanationHallucinationError` if the LLM cites any feature
  not in the SHAP top-5 input. No prompt alone — schema enforcement.
- **Cost tracking is mandatory**: `token_cost_usd=0.0` is rejected at the schema
  level. Every LLM call must provide real token counts.
- **Two audiences, one schema**: `target_audience: Literal["analyst", "customer"]`
  drives prompt selection. Customer mode strips fraud probability at the schema level.
- **Golden scenarios from day 0**: 10 evaluation scenarios defined before any
  implementation begins. They gate every phase.

## Tech Stack

- **Detection**: XGBoost + SHAP
- **Explanation**: LiteLLM (Haiku for generation, Sonnet for evaluation)
- **Structured output**: Instructor + Pydantic v2
- **Orchestration**: LangGraph
- **API**: FastAPI with SSE streaming
- **Observability**: Prometheus + Grafana
- **Testing**: Pytest + VCR.py (pytest-recording)
- **Deployment**: Docker Compose
