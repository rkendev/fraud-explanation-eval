"""Fraud detection result schema — output of XGBoost + SHAP."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))


class SHAPFeature(BaseModel):
    """Single SHAP feature attribution."""

    feature_name: str
    shap_value: float  # signed — positive pushes toward fraud
    feature_value: Any  # actual value from transaction


class FraudDetectionResult(BaseModel):
    """Output of DetectorModel: XGBoost prediction + SHAP attribution."""

    transaction_id: str
    fraud_probability: float
    is_fraud_predicted: bool
    top_shap_features: list[SHAPFeature]  # exactly top 5 by |shap_value|
    model_version: str
    inference_latency_ms: float
    confidence_tier: Literal["high", "medium", "low"]

    @field_validator("fraud_probability")
    @classmethod
    def probability_in_bounds(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"fraud_probability must be in [0,1], got {v}")
        return v

    @field_validator("top_shap_features")
    @classmethod
    def shap_features_max_five(cls, v: list[SHAPFeature]) -> list[SHAPFeature]:
        if len(v) > 5:
            raise ValueError(f"top_shap_features must have ≤5 entries, got {len(v)}")
        return v

    @model_validator(mode="after")
    def confidence_tier_consistent_with_probability(self) -> FraudDetectionResult:
        p = self.fraud_probability
        if p > 0.8 or p < 0.2:
            expected = "high"
        elif (0.6 <= p <= 0.8) or (0.2 <= p < 0.4):
            expected = "medium"
        else:
            expected = "low"
        if self.confidence_tier != expected:
            raise ValueError(
                f"confidence_tier '{self.confidence_tier}' inconsistent "
                f"with fraud_probability {p:.3f} (expected '{expected}')"
            )
        return self

    @model_validator(mode="after")
    def is_fraud_consistent_with_probability(self) -> FraudDetectionResult:
        expected = self.fraud_probability >= FRAUD_THRESHOLD
        if self.is_fraud_predicted != expected:
            raise ValueError(
                f"is_fraud_predicted={self.is_fraud_predicted} inconsistent "
                f"with fraud_probability={self.fraud_probability:.3f} "
                f"at threshold={FRAUD_THRESHOLD}"
            )
        return self
