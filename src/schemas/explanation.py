"""LLM explanation result schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class ExplanationResult(BaseModel):
    """Output of ExplanationAgent: grounded natural language explanation."""

    transaction_id: str
    target_audience: Literal["analyst", "customer"]
    fraud_probability: float  # MUST equal FraudDetectionResult.fraud_probability
    is_fraud_predicted: bool  # MUST equal FraudDetectionResult.is_fraud_predicted
    explanation_text: str  # ≤300 words
    cited_features: list[str]  # feature names mentioned in explanation_text
    uncited_features: list[str]  # top_shap features NOT mentioned (allowed)
    hallucinated_features: list[str]  # INVARIANT: must always be empty
    uncertainty_flag: bool
    uncertainty_disclosure: str | None = None
    explanation_generated: bool = True
    warning: str | None = None
    token_cost_usd: float
    generation_latency_seconds: float

    @field_validator("explanation_text")
    @classmethod
    def explanation_word_limit(cls, v: str) -> str:
        word_count = len(v.split())
        if word_count > 300:
            raise ValueError(f"explanation_text exceeds 300 words: {word_count}")
        return v

    @field_validator("hallucinated_features")
    @classmethod
    def no_hallucinated_features(cls, v: list[str]) -> list[str]:
        if v:
            raise ValueError(
                f"ExplanationHallucinationError: LLM cited features not in "
                f"SHAP input: {v}. This is a critical failure."
            )
        return v

    @model_validator(mode="after")
    def uncertainty_must_be_disclosed(self) -> ExplanationResult:
        if self.uncertainty_flag and self.explanation_generated:
            if not self.uncertainty_disclosure:
                raise ValueError(
                    "uncertainty_flag=True requires uncertainty_disclosure to be set"
                )
        return self

    @model_validator(mode="after")
    def customer_must_not_reveal_probability(self) -> ExplanationResult:
        if self.target_audience == "customer" and self.explanation_generated:
            prob_str = f"{self.fraud_probability:.0%}"
            raw_str = str(round(self.fraud_probability, 4))
            if prob_str in self.explanation_text or raw_str in self.explanation_text:
                raise ValueError(
                    "Customer explanation must not contain fraud probability value"
                )
        return self

    @field_validator("token_cost_usd")
    @classmethod
    def cost_must_be_real(cls, v: float) -> float:
        # Explicitly reject the 0.0 default — cost must be populated from LiteLLM
        if v == 0.0:
            raise ValueError(
                "token_cost_usd is 0.0 — this must be populated from actual "
                "LiteLLM token counts, not left as a default."
            )
        return v


class ExplanationEvalResult(BaseModel):
    """Output of EvalAgent: LLM-as-judge scoring of ExplanationResult."""

    transaction_id: str
    target_audience: Literal["analyst", "customer"]
    grounding_score: float  # 0.0–1.0: claims traceable to SHAP features?
    clarity_score: float  # 0.0–1.0: clear for target audience?
    completeness_score: float  # 0.0–1.0: top fraud signals covered?
    audience_appropriateness_score: float  # 0.0–1.0: correct register?
    uncertainty_handling_score: float | None = None  # null if not uncertain
    overall_score: float  # weighted average
    pass_threshold: float = 0.70
    passed: bool
    failure_reasons: list[str]
    token_cost_usd: float

    @field_validator(
        "grounding_score",
        "clarity_score",
        "completeness_score",
        "audience_appropriateness_score",
        "overall_score",
    )
    @classmethod
    def score_in_bounds(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be in [0,1], got {v}")
        return v

    @model_validator(mode="after")
    def passed_consistent_with_score(self) -> ExplanationEvalResult:
        expected = self.overall_score >= self.pass_threshold
        if self.passed != expected:
            raise ValueError(
                f"passed={self.passed} inconsistent with "
                f"overall_score={self.overall_score:.3f} at "
                f"threshold={self.pass_threshold}"
            )
        return self

    @field_validator("token_cost_usd")
    @classmethod
    def cost_must_be_real(cls, v: float) -> float:
        if v == 0.0:
            raise ValueError(
                "token_cost_usd must be populated from actual LiteLLM response"
            )
        return v
