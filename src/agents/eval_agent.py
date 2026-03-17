"""EvalAgent: LLM-as-judge scoring of ExplanationResult quality."""

from __future__ import annotations

import os
import time
from typing import Literal

import instructor
import litellm
from pydantic import BaseModel, field_validator

from src.schemas.detection import FraudDetectionResult
from src.schemas.explanation import ExplanationEvalResult, ExplanationResult
from src.utils.cost_tracker import record_agent_call
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = os.getenv("EVAL_MODEL", "claude-sonnet-4-6")
TIMEOUT_SECONDS = float(os.getenv("EVAL_TIMEOUT", "20.0"))

# Weights when uncertainty_handling_score is present
_WEIGHTS_WITH_UNCERTAINTY = {
    "grounding": 0.25,
    "clarity": 0.20,
    "completeness": 0.20,
    "audience_appropriateness": 0.15,
    "uncertainty_handling": 0.20,
}

# Weights when uncertainty_handling_score is absent
_WEIGHTS_WITHOUT_UNCERTAINTY = {
    "grounding": 0.30,
    "clarity": 0.25,
    "completeness": 0.25,
    "audience_appropriateness": 0.20,
}


class _LLMEvalOutput(BaseModel):
    """Internal structured output model for the LLM judge response."""

    grounding_score: float
    clarity_score: float
    completeness_score: float
    audience_appropriateness_score: float
    uncertainty_handling_score: float | None = None
    failure_reasons: list[str]

    @field_validator(
        "grounding_score",
        "clarity_score",
        "completeness_score",
        "audience_appropriateness_score",
    )
    @classmethod
    def score_in_bounds(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("uncertainty_handling_score")
    @classmethod
    def uncertainty_score_in_bounds(cls, v: float | None) -> float | None:
        if v is not None:
            return max(0.0, min(1.0, v))
        return v


class EvalAgent:
    """Scores ExplanationResult quality using an LLM-as-judge rubric.

    Evaluates grounding, clarity, completeness, audience appropriateness,
    and (when applicable) uncertainty handling. Records all calls to cost_log.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_retries: int = 2) -> None:
        self.model = model
        self.max_retries = max_retries
        self.client = instructor.from_litellm(litellm.completion)

    def evaluate(
        self,
        explanation_result: ExplanationResult,
        detection_result: FraudDetectionResult,
    ) -> ExplanationEvalResult:
        """Score an explanation against the detection ground truth.

        Args:
            explanation_result: The explanation to evaluate.
            detection_result: Ground truth fraud detection output.

        Returns:
            ExplanationEvalResult with all rubric scores.
        """
        target_audience = explanation_result.target_audience

        # Build rubric prompt
        system_msg, user_msg = self._build_prompt(
            explanation_result, detection_result, target_audience
        )

        # Determine whether to request uncertainty scoring
        evaluate_uncertainty = explanation_result.uncertainty_flag

        # Call LLM
        start = time.monotonic()
        try:
            llm_output, raw_completion = self.client.create_with_completion(
                response_model=_LLMEvalOutput,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_retries=self.max_retries,
                timeout=TIMEOUT_SECONDS,
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            # Rule 9: log cost even on failure — no exceptions
            record_agent_call(
                agent_name="EvalAgent",
                model=self.model,
                input_tokens=0,
                output_tokens=0,
                transaction_id=explanation_result.transaction_id,
                phase="phase_4",
                duration_seconds=elapsed,
            )
            if elapsed >= TIMEOUT_SECONDS or "timeout" in str(e).lower():
                logger.warning(
                    "eval_timeout",
                    extra={
                        "transaction_id": explanation_result.transaction_id,
                        "elapsed": elapsed,
                    },
                )
                raise EvalTimeoutError(
                    f"EvalAgent timed out after {elapsed:.1f}s"
                ) from e
            raise

        elapsed = time.monotonic() - start

        # Extract token usage
        usage = raw_completion.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        # Resolve uncertainty_handling_score
        uncertainty_score = (
            llm_output.uncertainty_handling_score if evaluate_uncertainty else None
        )

        # Calculate overall score before recording cost (needed for confidence/passed)
        overall_score = self._calculate_overall_score(
            grounding=llm_output.grounding_score,
            clarity=llm_output.clarity_score,
            completeness=llm_output.completeness_score,
            audience_appropriateness=llm_output.audience_appropriateness_score,
            uncertainty_handling=uncertainty_score,
        )

        passed = overall_score >= 0.70
        failure_reasons = llm_output.failure_reasons if not passed else []

        # Record cost — Rule 9: every agent call logs cost
        cost_usd = record_agent_call(
            agent_name="EvalAgent",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            transaction_id=explanation_result.transaction_id,
            phase="phase_4",
            duration_seconds=elapsed,
            confidence=round(overall_score, 4),
            passed=passed,
        )

        return ExplanationEvalResult(
            transaction_id=explanation_result.transaction_id,
            target_audience=target_audience,
            grounding_score=llm_output.grounding_score,
            clarity_score=llm_output.clarity_score,
            completeness_score=llm_output.completeness_score,
            audience_appropriateness_score=llm_output.audience_appropriateness_score,
            uncertainty_handling_score=uncertainty_score,
            overall_score=round(overall_score, 4),
            passed=passed,
            failure_reasons=failure_reasons,
            token_cost_usd=cost_usd,
        )

    # -- Prompt builders --

    def _build_prompt(
        self,
        explanation: ExplanationResult,
        detection: FraudDetectionResult,
        target_audience: Literal["analyst", "customer"],
    ) -> tuple[str, str]:
        shap_block = "\n".join(
            f"  - {f.feature_name}: shap_value={f.shap_value:.4f}, "
            f"value={f.feature_value}"
            for f in detection.top_shap_features
        )
        shap_names = [f.feature_name for f in detection.top_shap_features]

        uncertainty_rubric = ""
        if explanation.uncertainty_flag:
            uncertainty_rubric = (
                "\n5. **Uncertainty Handling** (uncertainty_handling_score: 0.0–1.0):\n"
                "   - The model confidence is LOW. The explanation MUST clearly disclose "
                "uncertainty.\n"
                "   - Score 1.0: uncertainty is prominently disclosed with appropriate caveats.\n"
                "   - Score 0.0: uncertainty is not mentioned at all despite low confidence.\n"
                "   - If uncertainty_disclosure is empty/missing, score 0.0."
            )

        if target_audience == "analyst":
            audience_criteria = (
                "   - Uses precise, technical language appropriate for fraud analysts.\n"
                "   - Does NOT use customer-facing language like 'your card', 'your account'.\n"
                "   - References fraud probability and SHAP contributions with values."
            )
        else:
            audience_criteria = (
                "   - Uses simple, empathetic language appropriate for cardholders.\n"
                "   - Does NOT use technical terms (SHAP, XGBoost, model, fraud_probability).\n"
                "   - Does NOT reveal any probability, score, or percentage.\n"
                "   - Refers to status as 'flagged for review' or similar."
            )

        system_msg = (
            "You are a fraud explanation quality evaluator. Given an explanation of a "
            "fraud detection result, score it on 4-5 dimensions using the rubric below.\n\n"
            "RUBRIC:\n"
            "1. **Grounding** (grounding_score: 0.0–1.0):\n"
            "   - Are ALL claims in the explanation traceable to the provided SHAP features?\n"
            "   - Score 1.0: every claim maps to a SHAP feature; no fabricated evidence.\n"
            "   - Score 0.0: explanation invents features or evidence not in the SHAP list.\n"
            "   - If cited_features contains items NOT in the SHAP feature list, score 0.0.\n\n"
            "2. **Clarity** (clarity_score: 0.0–1.0):\n"
            "   - Is the explanation clearly written and easy to understand for the "
            f"target audience ({target_audience})?\n"
            "   - Score 1.0: well-structured, logical flow, unambiguous.\n"
            "   - Score 0.0: confusing, contradictory, or incoherent.\n\n"
            "3. **Completeness** (completeness_score: 0.0–1.0):\n"
            "   - Does the explanation cover the most important fraud signals?\n"
            "   - Top SHAP features by |shap_value| should be discussed.\n"
            "   - Score 1.0: all major signals addressed.\n"
            "   - Score 0.0: critical signals omitted.\n\n"
            "4. **Audience Appropriateness** (audience_appropriateness_score: 0.0–1.0):\n"
            f"   Target audience: {target_audience}\n"
            f"{audience_criteria}\n"
            "   - Score 1.0: perfect register for the audience.\n"
            "   - Score 0.0: completely wrong register."
            f"{uncertainty_rubric}\n\n"
            "RULES:\n"
            "- Return scores as floats between 0.0 and 1.0.\n"
            "- In failure_reasons, list specific problems found (empty list if good).\n"
            "- You are scoring the explanation, not rewriting it.\n"
            "- Do NOT invent claims about what the explanation 'should have said' "
            "unless citing a specific SHAP feature that was omitted."
        )

        user_msg = (
            f"Transaction ID: {explanation.transaction_id}\n"
            f"Target audience: {target_audience}\n"
            f"Fraud probability: {detection.fraud_probability:.1%}\n"
            f"Is fraud predicted: {detection.is_fraud_predicted}\n"
            f"Confidence tier: {detection.confidence_tier}\n\n"
            f"SHAP features (ground truth, ordered by |shap_value|):\n{shap_block}\n"
            f"SHAP feature names: {shap_names}\n\n"
            f"--- EXPLANATION TO EVALUATE ---\n"
            f"{explanation.explanation_text}\n"
            f"--- END EXPLANATION ---\n\n"
            f"Cited features: {explanation.cited_features}\n"
            f"Uncited features: {explanation.uncited_features}\n"
            f"Hallucinated features: {explanation.hallucinated_features}\n"
            f"Uncertainty flag: {explanation.uncertainty_flag}\n"
            f"Uncertainty disclosure: {explanation.uncertainty_disclosure}"
        )

        return system_msg, user_msg

    @staticmethod
    def _calculate_overall_score(
        *,
        grounding: float,
        clarity: float,
        completeness: float,
        audience_appropriateness: float,
        uncertainty_handling: float | None,
    ) -> float:
        if uncertainty_handling is not None:
            w = _WEIGHTS_WITH_UNCERTAINTY
            return (
                grounding * w["grounding"]
                + clarity * w["clarity"]
                + completeness * w["completeness"]
                + audience_appropriateness * w["audience_appropriateness"]
                + uncertainty_handling * w["uncertainty_handling"]
            )
        else:
            w = _WEIGHTS_WITHOUT_UNCERTAINTY
            return (
                grounding * w["grounding"]
                + clarity * w["clarity"]
                + completeness * w["completeness"]
                + audience_appropriateness * w["audience_appropriateness"]
            )


class EvalTimeoutError(TimeoutError):
    """Raised when the EvalAgent LLM call exceeds the timeout."""
