"""ExplanationAgent: generates grounded NL explanations from fraud detection results."""

from __future__ import annotations

import os
import time
from typing import Any, Literal

import instructor
import litellm
from pydantic import BaseModel

from src.schemas.detection import FraudDetectionResult, SHAPFeature
from src.schemas.explanation import ExplanationResult
from src.security.sanitizer import sanitize_external_text
from src.utils.cost_tracker import record_agent_call
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = os.getenv("EXPLANATION_MODEL", "claude-haiku-4-5")
TIMEOUT_SECONDS = float(os.getenv("EXPLANATION_TIMEOUT", "15.0"))

# Fields whose values are user-controlled text and must be sanitized
_TEXT_VALUE_FEATURES = {"DeviceInfo", "P_emaildomain", "R_emaildomain"}


class ExplanationHallucinationError(ValueError):
    """Raised when the LLM cited features not present in SHAP input."""

    def __init__(self, hallucinated: list[str]) -> None:
        super().__init__(
            f"ExplanationHallucinationError: LLM cited features not in "
            f"SHAP input: {hallucinated}"
        )
        self.hallucinated = hallucinated


class ExplanationTimeoutError(TimeoutError):
    """Raised when the LLM call exceeds the timeout."""


class _LLMExplanationOutput(BaseModel):
    """Internal structured output model for the LLM response.

    The LLM only produces these fields; the agent computes the rest.
    """

    explanation_text: str
    cited_features: list[str]
    uncertainty_disclosure: str | None = None


class ExplanationAgent:
    """Generates grounded natural-language explanations of fraud detection results.

    Supports two audience modes (analyst, customer) with distinct prompts.
    All calls are logged via cost_tracker.record_agent_call().
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_retries: int = 2) -> None:
        self.model = model
        self.max_retries = max_retries
        self.client = instructor.from_litellm(litellm.completion)

    def explain(
        self,
        detection_result: FraudDetectionResult,
        target_audience: Literal["analyst", "customer"],
    ) -> ExplanationResult:
        """Generate an explanation for the given detection result.

        Args:
            detection_result: Validated FraudDetectionResult from the detector.
            target_audience: "analyst" or "customer" — drives prompt selection.

        Returns:
            ExplanationResult with all fields populated.

        Raises:
            ExplanationHallucinationError: If LLM cites features not in SHAP input.
            ExplanationTimeoutError: If LLM call exceeds timeout.
        """
        # Pre-condition: need at least 1 SHAP feature
        if not detection_result.top_shap_features:
            return self._empty_shap_result(detection_result, target_audience)

        # Build prompt based on audience
        if target_audience == "analyst":
            system_msg, user_msg = self._build_analyst_prompt(detection_result)
        else:
            system_msg, user_msg = self._build_customer_prompt(detection_result)

        # Call LLM with structured output
        start = time.monotonic()
        try:
            llm_output, raw_completion = self.client.create_with_completion(
                response_model=_LLMExplanationOutput,
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
            if elapsed >= TIMEOUT_SECONDS or "timeout" in str(e).lower():
                logger.warning(
                    "explanation_timeout",
                    extra={
                        "transaction_id": detection_result.transaction_id,
                        "elapsed": elapsed,
                    },
                )
                # Rule 9: Every agent call appends one record — no exceptions
                record_agent_call(
                    agent_name="ExplanationAgent",
                    model=self.model,
                    input_tokens=0,
                    output_tokens=0,
                    transaction_id=detection_result.transaction_id,
                    phase="phase_3",
                    duration_seconds=elapsed,
                )
                return self._timeout_result(detection_result, target_audience, elapsed)
            raise

        elapsed = time.monotonic() - start

        # Extract token usage
        usage = raw_completion.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        # Record cost BEFORE hallucination check — CLAUDE.md Rule 9:
        # "Every agent call appends one record to cost_log.jsonl — no exceptions"
        cost_usd = record_agent_call(
            agent_name="ExplanationAgent",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            transaction_id=detection_result.transaction_id,
            phase="phase_3",
            duration_seconds=elapsed,
        )

        # Hallucination detection (after cost is logged)
        shap_names = {f.feature_name for f in detection_result.top_shap_features}
        hallucinated = [f for f in llm_output.cited_features if f not in shap_names]
        uncited = [
            f.feature_name
            for f in detection_result.top_shap_features
            if f.feature_name not in llm_output.cited_features
        ]

        if hallucinated:
            raise ExplanationHallucinationError(hallucinated)

        # Build result
        uncertainty_flag = detection_result.confidence_tier == "low"

        return ExplanationResult(
            transaction_id=detection_result.transaction_id,
            target_audience=target_audience,
            fraud_probability=detection_result.fraud_probability,
            is_fraud_predicted=detection_result.is_fraud_predicted,
            explanation_text=llm_output.explanation_text,
            cited_features=llm_output.cited_features,
            uncited_features=uncited,
            hallucinated_features=[],
            uncertainty_flag=uncertainty_flag,
            uncertainty_disclosure=llm_output.uncertainty_disclosure,
            explanation_generated=True,
            token_cost_usd=cost_usd,
            generation_latency_seconds=round(elapsed, 3),
        )

    # -- Prompt builders --

    def _build_analyst_prompt(self, result: FraudDetectionResult) -> tuple[str, str]:
        features = self._sanitize_features(result.top_shap_features)
        feature_block = "\n".join(
            f"  - {f.feature_name}: shap_value={f.shap_value:.4f}, "
            f"value={f.feature_value}"
            for f in features
        )

        uncertainty_instruction = ""
        if result.confidence_tier == "low":
            uncertainty_instruction = (
                "\nIMPORTANT: The model confidence is LOW. You MUST disclose this "
                "uncertainty and note limited confidence in your explanation."
            )

        system_msg = (
            "You are a fraud analysis assistant. Given a fraud detection result with "
            "SHAP feature attributions, produce a concise explanation for a fraud analyst.\n\n"
            "RULES:\n"
            "- You may ONLY reference features in the provided SHAP feature list.\n"
            f"- State the fraud probability as {result.fraud_probability:.1%}.\n"
            "- Explain how each cited feature contributes to the fraud score.\n"
            "- Use precise, technical language appropriate for a fraud analyst.\n"
            "- Do NOT use customer-facing language like 'your card' or 'your account'.\n"
            "- Keep the explanation under 300 words.\n"
            "- In cited_features, list ONLY feature names you actually reference."
            f"{uncertainty_instruction}"
        )

        user_msg = (
            f"Transaction ID: {result.transaction_id}\n"
            f"Fraud probability: {result.fraud_probability:.1%}\n"
            f"Predicted fraud: {result.is_fraud_predicted}\n"
            f"Confidence tier: {result.confidence_tier}\n\n"
            f"SHAP feature attributions (top {len(features)} by |shap_value|):\n"
            f"{feature_block}"
        )

        return system_msg, user_msg

    def _build_customer_prompt(self, result: FraudDetectionResult) -> tuple[str, str]:
        # Top 3 by |shap_value|, no raw values, no probability
        features = self._sanitize_features(result.top_shap_features)
        top_3 = sorted(features, key=lambda f: abs(f.shap_value), reverse=True)[:3]
        feature_block = "\n".join(f"  - {f.feature_name}" for f in top_3)

        uncertainty_instruction = ""
        if result.confidence_tier == "low":
            uncertainty_instruction = (
                "\nIMPORTANT: We are not fully certain about this result. You MUST "
                "clearly communicate that this is under review and we cannot yet "
                "determine the final outcome."
            )

        system_msg = (
            "You are a customer communication assistant. Given a fraud review summary, "
            "produce a clear, reassuring explanation for the cardholder.\n\n"
            "RULES:\n"
            "- Do NOT state any probability, score, or percentage.\n"
            "- Do NOT use technical terminology (SHAP, XGBoost, model, fraud_probability).\n"
            "- Use simple, empathetic language.\n"
            "- Refer to the transaction status as 'flagged for review' or 'held for review'.\n"
            "- Keep the explanation under 300 words.\n"
            "- In cited_features, list ONLY feature names you actually reference."
            f"{uncertainty_instruction}"
        )

        status = "flagged for review" if result.is_fraud_predicted else "appears normal"
        user_msg = (
            f"Transaction {result.transaction_id} has been {status}.\n\n"
            f"Factors considered:\n{feature_block}"
        )

        return system_msg, user_msg

    # -- Helpers --

    def _sanitize_features(self, features: list[SHAPFeature]) -> list[SHAPFeature]:
        """Sanitize string-valued feature values before they enter the prompt."""
        sanitized = []
        for f in features:
            if f.feature_name in _TEXT_VALUE_FEATURES and isinstance(
                f.feature_value, str
            ):
                clean_value: Any = sanitize_external_text(
                    f.feature_value, source=f.feature_name
                )
                sanitized.append(
                    SHAPFeature(
                        feature_name=f.feature_name,
                        shap_value=f.shap_value,
                        feature_value=clean_value,
                    )
                )
            else:
                sanitized.append(f)
        return sanitized

    def _empty_shap_result(
        self,
        result: FraudDetectionResult,
        target_audience: Literal["analyst", "customer"],
    ) -> ExplanationResult:
        return ExplanationResult(
            transaction_id=result.transaction_id,
            target_audience=target_audience,
            fraud_probability=result.fraud_probability,
            is_fraud_predicted=result.is_fraud_predicted,
            explanation_text="",
            cited_features=[],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=result.confidence_tier == "low",
            explanation_generated=False,
            warning="insufficient_shap_data",
            token_cost_usd=0.0,
            generation_latency_seconds=0.0,
        )

    def _timeout_result(
        self,
        result: FraudDetectionResult,
        target_audience: Literal["analyst", "customer"],
        elapsed: float,
    ) -> ExplanationResult:
        return ExplanationResult(
            transaction_id=result.transaction_id,
            target_audience=target_audience,
            fraud_probability=result.fraud_probability,
            is_fraud_predicted=result.is_fraud_predicted,
            explanation_text="",
            cited_features=[],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=result.confidence_tier == "low",
            explanation_generated=False,
            warning="llm_timeout",
            token_cost_usd=0.0,
            generation_latency_seconds=round(elapsed, 3),
        )
