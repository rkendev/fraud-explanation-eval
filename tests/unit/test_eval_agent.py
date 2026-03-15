"""Tests for EvalAgent — LLM-as-judge scoring of ExplanationResult."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agents.eval_agent import (
    _WEIGHTS_WITH_UNCERTAINTY,
    _WEIGHTS_WITHOUT_UNCERTAINTY,
    EvalAgent,
    EvalTimeoutError,
    _LLMEvalOutput,
)
from src.schemas.detection import FraudDetectionResult, SHAPFeature
from src.schemas.explanation import ExplanationEvalResult, ExplanationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(
    *,
    transaction_id: str = "TX_EVAL_001",
    fraud_probability: float = 0.87,
    is_fraud_predicted: bool = True,
    confidence_tier: str = "high",
    num_features: int = 5,
) -> FraudDetectionResult:
    # Ensure fraud_probability is consistent with confidence_tier
    if confidence_tier == "low":
        fraud_probability = 0.52
        is_fraud_predicted = True
    elif confidence_tier == "medium" and fraud_probability > 0.8:
        fraud_probability = 0.65
    features = [
        SHAPFeature(
            feature_name="TransactionAmt", shap_value=0.45, feature_value=299.99
        ),
        SHAPFeature(
            feature_name="DeviceInfo", shap_value=0.31, feature_value="Windows 10"
        ),
        SHAPFeature(
            feature_name="P_emaildomain", shap_value=0.22, feature_value="gmail.com"
        ),
        SHAPFeature(feature_name="card6", shap_value=0.18, feature_value="debit"),
        SHAPFeature(feature_name="addr1", shap_value=-0.12, feature_value=325),
    ][:num_features]
    return FraudDetectionResult(
        transaction_id=transaction_id,
        fraud_probability=fraud_probability,
        is_fraud_predicted=is_fraud_predicted,
        top_shap_features=features,
        model_version="1.0.0",
        inference_latency_ms=4.2,
        confidence_tier=confidence_tier,
    )


def _make_explanation(
    *,
    transaction_id: str = "TX_EVAL_001",
    target_audience: str = "analyst",
    fraud_probability: float = 0.87,
    is_fraud_predicted: bool = True,
    explanation_text: str = (
        "Transaction TX_EVAL_001 has a fraud probability of 87%. "
        "The TransactionAmt of $299.99 contributed most (SHAP 0.45). "
        "DeviceInfo and P_emaildomain also contributed to the score."
    ),
    cited_features: list[str] | None = None,
    uncited_features: list[str] | None = None,
    uncertainty_flag: bool = False,
    uncertainty_disclosure: str | None = None,
) -> ExplanationResult:
    if cited_features is None:
        cited_features = ["TransactionAmt", "DeviceInfo", "P_emaildomain"]
    if uncited_features is None:
        uncited_features = ["card6", "addr1"]
    return ExplanationResult(
        transaction_id=transaction_id,
        target_audience=target_audience,
        fraud_probability=fraud_probability,
        is_fraud_predicted=is_fraud_predicted,
        explanation_text=explanation_text,
        cited_features=cited_features,
        uncited_features=uncited_features,
        hallucinated_features=[],
        uncertainty_flag=uncertainty_flag,
        uncertainty_disclosure=uncertainty_disclosure,
        explanation_generated=True,
        token_cost_usd=0.001,
        generation_latency_seconds=2.5,
    )


def _make_llm_output(
    *,
    grounding: float = 0.95,
    clarity: float = 0.90,
    completeness: float = 0.85,
    audience: float = 0.90,
    uncertainty: float | None = None,
    failure_reasons: list[str] | None = None,
) -> _LLMEvalOutput:
    return _LLMEvalOutput(
        grounding_score=grounding,
        clarity_score=clarity,
        completeness_score=completeness,
        audience_appropriateness_score=audience,
        uncertainty_handling_score=uncertainty,
        failure_reasons=failure_reasons or [],
    )


def _mock_raw_completion(input_tokens: int = 1200, output_tokens: int = 300) -> Any:
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
    )


def _patch_llm(
    llm_output: _LLMEvalOutput,
    raw_completion: Any | None = None,
):
    """Return a patch context for instructor client.create_with_completion."""
    if raw_completion is None:
        raw_completion = _mock_raw_completion()
    return patch(
        "src.agents.eval_agent.instructor.from_litellm",
        return_value=MagicMock(
            create_with_completion=MagicMock(return_value=(llm_output, raw_completion))
        ),
    )


# ===========================================================================
# 1. Agent Initialization
# ===========================================================================


class TestEvalAgentInit:
    def test_default_model(self):
        agent = EvalAgent()
        assert agent.model == "claude-sonnet-4-6"

    def test_custom_model(self):
        agent = EvalAgent(model="gpt-4o")
        assert agent.model == "gpt-4o"

    def test_custom_retries(self):
        agent = EvalAgent(max_retries=5)
        assert agent.max_retries == 5


# ===========================================================================
# 2. High-Score Explanation Passes
# ===========================================================================


class TestHighScoreExplanation:
    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_high_score_passes(self, mock_cost):
        llm_out = _make_llm_output(
            grounding=0.95, clarity=0.90, completeness=0.85, audience=0.90
        )
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(_make_explanation(), _make_detection())
        assert result.passed is True
        assert result.overall_score >= 0.70
        assert result.failure_reasons == []

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_all_perfect_scores(self, mock_cost):
        llm_out = _make_llm_output(
            grounding=1.0, clarity=1.0, completeness=1.0, audience=1.0
        )
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(_make_explanation(), _make_detection())
        assert result.passed is True
        assert result.overall_score == 1.0

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_result_schema_valid(self, mock_cost):
        llm_out = _make_llm_output()
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(_make_explanation(), _make_detection())
        assert isinstance(result, ExplanationEvalResult)
        assert result.transaction_id == "TX_EVAL_001"
        assert result.target_audience == "analyst"


# ===========================================================================
# 3. Low-Score Explanation Fails
# ===========================================================================


class TestLowScoreExplanation:
    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_low_scores_fail(self, mock_cost):
        llm_out = _make_llm_output(
            grounding=0.3,
            clarity=0.4,
            completeness=0.2,
            audience=0.5,
            failure_reasons=["Claims not grounded in SHAP", "Missing key features"],
        )
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(_make_explanation(), _make_detection())
        assert result.passed is False
        assert result.overall_score < 0.70
        assert len(result.failure_reasons) > 0

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_borderline_fail(self, mock_cost):
        # Just below threshold
        llm_out = _make_llm_output(
            grounding=0.69,
            clarity=0.69,
            completeness=0.69,
            audience=0.69,
            failure_reasons=["Borderline quality"],
        )
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(_make_explanation(), _make_detection())
        assert result.passed is False
        assert result.overall_score < 0.70

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_poor_explanation_text(self, mock_cost):
        poor_explanation = _make_explanation(
            explanation_text="Transaction flagged. Could be fraud.",
            cited_features=["TransactionAmt"],
            uncited_features=["DeviceInfo", "P_emaildomain", "card6", "addr1"],
        )
        llm_out = _make_llm_output(
            grounding=0.4,
            clarity=0.3,
            completeness=0.2,
            audience=0.5,
            failure_reasons=[
                "Explanation lacks detail",
                "Does not explain SHAP contributions",
            ],
        )
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(poor_explanation, _make_detection())
        assert result.passed is False


# ===========================================================================
# 4. Customer Explanation Fails Analyst Rubric
# ===========================================================================


class TestAudienceMismatch:
    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_customer_explanation_scored_correctly(self, mock_cost):
        customer_expl = _make_explanation(
            target_audience="customer",
            explanation_text=(
                "Your recent transaction has been held for review. "
                "We noticed unusual activity related to the transaction amount "
                "and the device used."
            ),
            cited_features=["TransactionAmt", "DeviceInfo"],
            uncited_features=["P_emaildomain", "card6", "addr1"],
        )
        llm_out = _make_llm_output(
            grounding=0.90,
            clarity=0.85,
            completeness=0.70,
            audience=0.95,
        )
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(customer_expl, _make_detection())
        assert result.target_audience == "customer"
        assert result.passed is True

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_customer_with_analyst_language_fails(self, mock_cost):
        bad_customer = _make_explanation(
            target_audience="customer",
            explanation_text=(
                "The XGBoost model SHAP analysis shows TransactionAmt has "
                "a shap_value of 0.45. The fraud_probability is elevated."
            ),
            cited_features=["TransactionAmt"],
            uncited_features=["DeviceInfo", "P_emaildomain", "card6", "addr1"],
        )
        llm_out = _make_llm_output(
            grounding=0.80,
            clarity=0.50,
            completeness=0.40,
            audience=0.10,
            failure_reasons=[
                "Uses technical terms (XGBoost, SHAP, shap_value, fraud_probability)",
                "Not appropriate for customer audience",
            ],
        )
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(bad_customer, _make_detection())
        assert result.passed is False
        assert result.audience_appropriateness_score <= 0.30


# ===========================================================================
# 5. Uncertainty Not Disclosed Fails
# ===========================================================================


class TestUncertaintyHandling:
    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_uncertainty_disclosed_passes(self, mock_cost):
        uncertain_expl = _make_explanation(
            fraud_probability=0.52,
            uncertainty_flag=True,
            uncertainty_disclosure="Model confidence is low; result under review.",
            explanation_text=(
                "Transaction TX_EVAL_001 has a fraud probability of 52%. "
                "The TransactionAmt contributed most. "
                "Note: model confidence is limited."
            ),
        )
        detection = _make_detection(confidence_tier="low")
        llm_out = _make_llm_output(
            grounding=0.90,
            clarity=0.85,
            completeness=0.80,
            audience=0.85,
            uncertainty=0.90,
        )
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(uncertain_expl, detection)
        assert result.uncertainty_handling_score == 0.90
        assert result.passed is True

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_uncertainty_not_disclosed_fails(self, mock_cost):
        uncertain_expl = _make_explanation(
            fraud_probability=0.52,
            uncertainty_flag=True,
            uncertainty_disclosure="Limited confidence.",
            explanation_text=(
                "Transaction TX_EVAL_001 has a fraud probability of 52%. "
                "The TransactionAmt contributed most."
            ),
        )
        detection = _make_detection(confidence_tier="low")
        llm_out = _make_llm_output(
            grounding=0.85,
            clarity=0.80,
            completeness=0.70,
            audience=0.80,
            uncertainty=0.10,
            failure_reasons=[
                "Uncertainty not adequately disclosed despite low confidence"
            ],
        )
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(uncertain_expl, detection)
        assert result.uncertainty_handling_score == 0.10
        assert result.passed is False

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_no_uncertainty_flag_skips_score(self, mock_cost):
        llm_out = _make_llm_output(uncertainty=None)
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(
                _make_explanation(uncertainty_flag=False), _make_detection()
            )
        assert result.uncertainty_handling_score is None

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_uncertainty_flag_forces_scoring(self, mock_cost):
        uncertain_expl = _make_explanation(
            fraud_probability=0.52,
            uncertainty_flag=True,
            uncertainty_disclosure="Low confidence result.",
            explanation_text=(
                "Transaction TX_EVAL_001 has been flagged. "
                "The TransactionAmt contributed most. "
                "Note: confidence is limited."
            ),
        )
        detection = _make_detection(confidence_tier="low")
        llm_out = _make_llm_output(uncertainty=0.75)
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(uncertain_expl, detection)
        assert result.uncertainty_handling_score is not None


# ===========================================================================
# 6. Hallucinated Feature Fails Grounding Score
# ===========================================================================


class TestGroundingFailure:
    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_grounding_zero_for_bad_explanation(self, mock_cost):
        llm_out = _make_llm_output(
            grounding=0.0,
            clarity=0.60,
            completeness=0.50,
            audience=0.70,
            failure_reasons=["Explanation references features not in SHAP list"],
        )
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(_make_explanation(), _make_detection())
        assert result.grounding_score == 0.0
        assert result.passed is False

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_partial_grounding(self, mock_cost):
        llm_out = _make_llm_output(
            grounding=0.50,
            clarity=0.80,
            completeness=0.70,
            audience=0.80,
            failure_reasons=["Some claims not traceable to SHAP features"],
        )
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(_make_explanation(), _make_detection())
        assert result.grounding_score == 0.50


# ===========================================================================
# 7. Cost Tracking
# ===========================================================================


class TestCostTracking:
    @patch("src.agents.eval_agent.record_agent_call", return_value=0.0189)
    def test_cost_recorded(self, mock_cost):
        llm_out = _make_llm_output()
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(_make_explanation(), _make_detection())
        mock_cost.assert_called_once()
        call_kwargs = mock_cost.call_args.kwargs
        assert call_kwargs["agent_name"] == "EvalAgent"
        assert call_kwargs["phase"] == "phase_4"
        assert call_kwargs["transaction_id"] == "TX_EVAL_001"
        assert result.token_cost_usd == 0.0189

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_cost_uses_actual_tokens(self, mock_cost):
        llm_out = _make_llm_output()
        raw = _mock_raw_completion(input_tokens=1500, output_tokens=400)
        with _patch_llm(llm_out, raw):
            agent = EvalAgent()
            agent.evaluate(_make_explanation(), _make_detection())
        call_kwargs = mock_cost.call_args.kwargs
        assert call_kwargs["input_tokens"] == 1500
        assert call_kwargs["output_tokens"] == 400

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_cost_uses_model_name(self, mock_cost):
        llm_out = _make_llm_output()
        with _patch_llm(llm_out):
            agent = EvalAgent(model="gpt-4o")
            agent.evaluate(_make_explanation(), _make_detection())
        assert mock_cost.call_args.kwargs["model"] == "gpt-4o"


# ===========================================================================
# 8. Timeout Handling
# ===========================================================================


class TestTimeoutHandling:
    @patch("src.agents.eval_agent.record_agent_call", return_value=0.0)
    def test_timeout_raises_eval_timeout_error(self, mock_cost):
        with patch(
            "src.agents.eval_agent.instructor.from_litellm",
            return_value=MagicMock(
                create_with_completion=MagicMock(
                    side_effect=Exception("Request timeout")
                )
            ),
        ), patch("src.agents.eval_agent.TIMEOUT_SECONDS", 0.001):
            agent = EvalAgent()
            with pytest.raises(EvalTimeoutError):
                agent.evaluate(_make_explanation(), _make_detection())

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.0)
    def test_non_timeout_exception_propagates(self, mock_cost):
        with patch(
            "src.agents.eval_agent.instructor.from_litellm",
            return_value=MagicMock(
                create_with_completion=MagicMock(
                    side_effect=ValueError("Invalid model parameter")
                )
            ),
        ), patch("src.agents.eval_agent.time.monotonic", side_effect=[0.0, 1.0]):
            agent = EvalAgent()
            with pytest.raises(ValueError, match="Invalid model parameter"):
                agent.evaluate(_make_explanation(), _make_detection())

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.0)
    def test_timeout_still_logs_cost(self, mock_cost):
        """Rule 9: cost must be logged even on timeout."""
        with patch(
            "src.agents.eval_agent.instructor.from_litellm",
            return_value=MagicMock(
                create_with_completion=MagicMock(
                    side_effect=Exception("Request timeout")
                )
            ),
        ), patch("src.agents.eval_agent.TIMEOUT_SECONDS", 0.001):
            agent = EvalAgent()
            with pytest.raises(EvalTimeoutError):
                agent.evaluate(_make_explanation(), _make_detection())
        mock_cost.assert_called_once()
        call_kwargs = mock_cost.call_args.kwargs
        assert call_kwargs["agent_name"] == "EvalAgent"
        assert call_kwargs["input_tokens"] == 0
        assert call_kwargs["output_tokens"] == 0

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.0)
    def test_non_timeout_exception_still_logs_cost(self, mock_cost):
        """Rule 9: cost must be logged even on non-timeout errors."""
        with patch(
            "src.agents.eval_agent.instructor.from_litellm",
            return_value=MagicMock(
                create_with_completion=MagicMock(
                    side_effect=ValueError("Invalid model parameter")
                )
            ),
        ), patch("src.agents.eval_agent.time.monotonic", side_effect=[0.0, 1.0]):
            agent = EvalAgent()
            with pytest.raises(ValueError, match="Invalid model parameter"):
                agent.evaluate(_make_explanation(), _make_detection())
        mock_cost.assert_called_once()


# ===========================================================================
# 9. Overall Score Calculation
# ===========================================================================


class TestOverallScoreCalculation:
    def test_weighted_average_without_uncertainty(self):
        score = EvalAgent._calculate_overall_score(
            grounding=1.0,
            clarity=1.0,
            completeness=1.0,
            audience_appropriateness=1.0,
            uncertainty_handling=None,
        )
        assert score == pytest.approx(1.0)

    def test_weighted_average_with_uncertainty(self):
        score = EvalAgent._calculate_overall_score(
            grounding=1.0,
            clarity=1.0,
            completeness=1.0,
            audience_appropriateness=1.0,
            uncertainty_handling=1.0,
        )
        assert score == pytest.approx(1.0)

    def test_zero_scores(self):
        score = EvalAgent._calculate_overall_score(
            grounding=0.0,
            clarity=0.0,
            completeness=0.0,
            audience_appropriateness=0.0,
            uncertainty_handling=None,
        )
        assert score == pytest.approx(0.0)

    def test_mixed_scores_without_uncertainty(self):
        w = _WEIGHTS_WITHOUT_UNCERTAINTY
        expected = (
            0.8 * w["grounding"]
            + 0.7 * w["clarity"]
            + 0.6 * w["completeness"]
            + 0.9 * w["audience_appropriateness"]
        )
        score = EvalAgent._calculate_overall_score(
            grounding=0.8,
            clarity=0.7,
            completeness=0.6,
            audience_appropriateness=0.9,
            uncertainty_handling=None,
        )
        assert score == pytest.approx(expected)

    def test_mixed_scores_with_uncertainty(self):
        w = _WEIGHTS_WITH_UNCERTAINTY
        expected = (
            0.8 * w["grounding"]
            + 0.7 * w["clarity"]
            + 0.6 * w["completeness"]
            + 0.9 * w["audience_appropriateness"]
            + 0.5 * w["uncertainty_handling"]
        )
        score = EvalAgent._calculate_overall_score(
            grounding=0.8,
            clarity=0.7,
            completeness=0.6,
            audience_appropriateness=0.9,
            uncertainty_handling=0.5,
        )
        assert score == pytest.approx(expected)

    def test_weights_without_uncertainty_sum_to_one(self):
        assert sum(_WEIGHTS_WITHOUT_UNCERTAINTY.values()) == pytest.approx(1.0)

    def test_weights_with_uncertainty_sum_to_one(self):
        assert sum(_WEIGHTS_WITH_UNCERTAINTY.values()) == pytest.approx(1.0)


# ===========================================================================
# 10. Prompt Construction
# ===========================================================================


class TestPromptConstruction:
    def test_analyst_prompt_contains_shap_values(self):
        agent = EvalAgent()
        detection = _make_detection()
        explanation = _make_explanation()
        system_msg, user_msg = agent._build_prompt(explanation, detection, "analyst")
        assert "shap_value=0.4500" in user_msg
        assert "TransactionAmt" in user_msg
        assert "technical language" in system_msg

    def test_customer_prompt_contains_audience(self):
        agent = EvalAgent()
        detection = _make_detection()
        explanation = _make_explanation(
            target_audience="customer",
            explanation_text=(
                "Your recent transaction has been held for review. "
                "We noticed unusual activity related to the transaction amount "
                "and the device used."
            ),
            cited_features=["TransactionAmt", "DeviceInfo"],
            uncited_features=["P_emaildomain", "card6", "addr1"],
        )
        system_msg, user_msg = agent._build_prompt(explanation, detection, "customer")
        assert "customer" in system_msg.lower()
        assert "simple, empathetic" in system_msg

    def test_uncertainty_rubric_added_when_flagged(self):
        agent = EvalAgent()
        detection = _make_detection(confidence_tier="low")
        explanation = _make_explanation(
            fraud_probability=0.52,
            uncertainty_flag=True,
            uncertainty_disclosure="Low confidence.",
            explanation_text=(
                "Transaction TX_EVAL_001 has been flagged. "
                "The TransactionAmt contributed most. "
                "Note: confidence is limited."
            ),
        )
        system_msg, _ = agent._build_prompt(explanation, detection, "analyst")
        assert "Uncertainty Handling" in system_msg
        assert "uncertainty_handling_score" in system_msg

    def test_uncertainty_rubric_absent_when_not_flagged(self):
        agent = EvalAgent()
        detection = _make_detection()
        explanation = _make_explanation(uncertainty_flag=False)
        system_msg, _ = agent._build_prompt(explanation, detection, "analyst")
        assert "Uncertainty Handling" not in system_msg

    def test_prompt_includes_explanation_text(self):
        agent = EvalAgent()
        detection = _make_detection()
        explanation = _make_explanation()
        _, user_msg = agent._build_prompt(explanation, detection, "analyst")
        assert explanation.explanation_text in user_msg

    def test_prompt_includes_cited_features(self):
        agent = EvalAgent()
        detection = _make_detection()
        explanation = _make_explanation()
        _, user_msg = agent._build_prompt(explanation, detection, "analyst")
        assert "TransactionAmt" in user_msg
        assert "DeviceInfo" in user_msg


# ===========================================================================
# 11. Schema Integration
# ===========================================================================


class TestSchemaIntegration:
    def test_eval_result_rejects_inconsistent_passed(self):
        with pytest.raises(ValueError, match="inconsistent"):
            ExplanationEvalResult(
                transaction_id="TX_001",
                target_audience="analyst",
                grounding_score=0.9,
                clarity_score=0.9,
                completeness_score=0.9,
                audience_appropriateness_score=0.9,
                overall_score=0.9,
                passed=False,  # inconsistent with overall_score >= 0.70
                failure_reasons=[],
                token_cost_usd=0.01,
            )

    def test_eval_result_rejects_zero_cost(self):
        with pytest.raises(ValueError, match="token_cost_usd"):
            ExplanationEvalResult(
                transaction_id="TX_001",
                target_audience="analyst",
                grounding_score=0.9,
                clarity_score=0.9,
                completeness_score=0.9,
                audience_appropriateness_score=0.9,
                overall_score=0.9,
                passed=True,
                failure_reasons=[],
                token_cost_usd=0.0,
            )

    def test_eval_result_rejects_out_of_bounds_score(self):
        with pytest.raises(ValueError, match="Score must be"):
            ExplanationEvalResult(
                transaction_id="TX_001",
                target_audience="analyst",
                grounding_score=1.5,
                clarity_score=0.9,
                completeness_score=0.9,
                audience_appropriateness_score=0.9,
                overall_score=0.9,
                passed=True,
                failure_reasons=[],
                token_cost_usd=0.01,
            )

    @patch("src.agents.eval_agent.record_agent_call", return_value=0.015)
    def test_failure_reasons_empty_on_pass(self, mock_cost):
        llm_out = _make_llm_output(failure_reasons=["Should be cleared on pass"])
        with _patch_llm(llm_out):
            agent = EvalAgent()
            result = agent.evaluate(_make_explanation(), _make_detection())
        assert result.passed is True
        assert result.failure_reasons == []


# ===========================================================================
# 12. Internal LLM Output Validation
# ===========================================================================


class TestLLMOutputValidation:
    def test_scores_clamped_to_bounds(self):
        out = _LLMEvalOutput(
            grounding_score=1.5,
            clarity_score=-0.1,
            completeness_score=0.5,
            audience_appropriateness_score=2.0,
            failure_reasons=[],
        )
        assert out.grounding_score == 1.0
        assert out.clarity_score == 0.0
        assert out.audience_appropriateness_score == 1.0

    def test_uncertainty_score_clamped(self):
        out = _LLMEvalOutput(
            grounding_score=0.5,
            clarity_score=0.5,
            completeness_score=0.5,
            audience_appropriateness_score=0.5,
            uncertainty_handling_score=1.5,
            failure_reasons=[],
        )
        assert out.uncertainty_handling_score == 1.0

    def test_none_uncertainty_allowed(self):
        out = _LLMEvalOutput(
            grounding_score=0.5,
            clarity_score=0.5,
            completeness_score=0.5,
            audience_appropriateness_score=0.5,
            uncertainty_handling_score=None,
            failure_reasons=[],
        )
        assert out.uncertainty_handling_score is None
