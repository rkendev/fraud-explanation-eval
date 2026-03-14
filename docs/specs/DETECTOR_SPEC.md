# Agent Behavior Specification — DetectorModel

## Identity
- Component: DetectorModel (XGBoost classifier, no LLM)
- Role: Produce fraud probability + SHAP-grounded feature attribution
- Model: XGBoost binary classifier, trained on IEEE-CIS
- Max execution time: 2 seconds (inference only, model pre-loaded)

## Input Contract
Schema: FraudTransaction
- TransactionID: str (required, must be non-empty)
- TransactionAmt: float (required, > 0.0)
- ProductCD: Literal["W","H","C","S","R"] (required)
- card1: int (1000–18396)
- card4: Optional[Literal["discover","mastercard","visa","american express"]]
- card6: Optional[Literal["credit","debit"]]
- addr1: Optional[int]
- P_emaildomain: Optional[str]
- R_emaildomain: Optional[str]
- DeviceType: Optional[Literal["desktop","mobile"]]
- DeviceInfo: Optional[str]
Pre-conditions: TransactionID must be unique within a batch run.

## Output Contract
Schema: FraudDetectionResult
- transaction_id: str
- fraud_probability: float (0.0–1.0)
- is_fraud_predicted: bool (threshold: 0.5, configurable via env)
- top_shap_features: list[SHAPFeature] (exactly top 5 by |shap_value|)
  - SHAPFeature: {feature_name: str, shap_value: float, feature_value: Any}
- model_version: str (semver, loaded from models/artifacts/version.txt)
- inference_latency_ms: float
- confidence_tier: Literal["high","medium","low"]
  - high: probability > 0.8 or < 0.2
  - medium: 0.6–0.8 or 0.2–0.4
  - low: 0.4–0.6 (uncertain — must propagate to ExplanationResult)

## Failure Modes
| Failure | Trigger | Response | Downstream Impact |
|---------|---------|----------|-------------------|
| Model not loaded | File missing | Raise ModelNotLoadedError | Pipeline halts |
| Feature validation fail | Invalid field value | Raise TransactionValidationError | Pipeline halts, log |
| Inference timeout | >2s | Raise InferenceTimeoutError | Return with is_fraud_predicted=None |
| SHAP computation fail | SHAP library error | Return result with top_shap_features=[] and warning | ExplanationAgent must handle empty SHAP |

## Hallucination Constraints
- Not applicable (no LLM involved)
- SHAP values are deterministic for a given model version and input

## Confidence Threshold Rationale
- fraud_probability > 0.5 = fraud predicted (standard binary classification)
- confidence_tier "low" (0.4–0.6) triggers ExplanationResult.uncertainty_flag=True
- Test asserting this: tests/unit/test_schemas.py::test_confidence_tier_boundaries

## Phase Gate Requirements
- Branch coverage ≥ 85% on src/models/
- Required tests: valid inference, all confidence tiers, all failure modes,
  SHAP output structure, model version loading
