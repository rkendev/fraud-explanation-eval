"""Adversarial data leakage tests.

Tests verify that customer-facing explanations do not leak sensitive
data that should only be available to analysts:
- Raw fraud probability values
- SHAP values and technical terminology
- Internal model details
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError  # noqa: F401

from src.schemas.explanation import ExplanationResult

# ---------------------------------------------------------------------------
# Customer probability leakage — schema enforced
# ---------------------------------------------------------------------------


class TestCustomerProbabilityLeakage:
    """Customer explanations must not contain the fraud probability value."""

    def test_percent_format_rejected(self):
        """'87%' in customer text is rejected."""
        with pytest.raises(ValidationError, match="must not contain fraud probability"):
            ExplanationResult(
                transaction_id="TX_LEAK_001",
                target_audience="customer",
                fraud_probability=0.87,
                is_fraud_predicted=True,
                explanation_text="We detected an 87% chance of fraud on your account.",
                cited_features=["TransactionAmt"],
                uncited_features=[],
                hallucinated_features=[],
                uncertainty_flag=False,
                explanation_generated=True,
                token_cost_usd=0.001,
                generation_latency_seconds=2.0,
            )

    def test_raw_float_rejected(self):
        """'0.87' in customer text is rejected."""
        with pytest.raises(ValidationError, match="must not contain fraud probability"):
            ExplanationResult(
                transaction_id="TX_LEAK_002",
                target_audience="customer",
                fraud_probability=0.87,
                is_fraud_predicted=True,
                explanation_text="The fraud score of 0.87 triggered this alert.",
                cited_features=[],
                uncited_features=[],
                hallucinated_features=[],
                uncertainty_flag=False,
                explanation_generated=True,
                token_cost_usd=0.001,
                generation_latency_seconds=2.0,
            )

    def test_low_probability_format_rejected(self):
        """Even low probability values must not appear in customer text."""
        with pytest.raises(ValidationError, match="must not contain fraud probability"):
            ExplanationResult(
                transaction_id="TX_LEAK_003",
                target_audience="customer",
                fraud_probability=0.15,
                is_fraud_predicted=False,
                explanation_text="Your transaction scored 0.15 on our fraud scale.",
                cited_features=[],
                uncited_features=[],
                hallucinated_features=[],
                uncertainty_flag=False,
                explanation_generated=True,
                token_cost_usd=0.001,
                generation_latency_seconds=2.0,
            )

    def test_borderline_probability_rejected(self):
        """Borderline probability (0.5) must not appear in customer text."""
        with pytest.raises(ValidationError, match="must not contain fraud probability"):
            ExplanationResult(
                transaction_id="TX_LEAK_004",
                target_audience="customer",
                fraud_probability=0.50,
                is_fraud_predicted=True,
                explanation_text="The model assigned a score of 0.5 to this transaction.",
                cited_features=[],
                uncited_features=[],
                hallucinated_features=[],
                uncertainty_flag=False,
                explanation_generated=True,
                token_cost_usd=0.001,
                generation_latency_seconds=2.0,
            )

    def test_safe_customer_text_accepted(self):
        """Customer text without probability is accepted."""
        result = ExplanationResult(
            transaction_id="TX_LEAK_005",
            target_audience="customer",
            fraud_probability=0.87,
            is_fraud_predicted=True,
            explanation_text=(
                "Your recent transaction has been held for review. "
                "We noticed some unusual activity with the transaction amount."
            ),
            cited_features=["TransactionAmt"],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=False,
            explanation_generated=True,
            token_cost_usd=0.001,
            generation_latency_seconds=2.0,
        )
        assert result.target_audience == "customer"
        assert "87" not in result.explanation_text
        assert "0.87" not in result.explanation_text


# ---------------------------------------------------------------------------
# Analyst mode is allowed to contain SHAP values (positive test)
# ---------------------------------------------------------------------------


class TestAnalystModeAllowsSHAP:
    """Analyst explanations may contain probability and SHAP data."""

    def test_analyst_can_state_probability(self):
        """Analyst mode is allowed to include the raw probability."""
        result = ExplanationResult(
            transaction_id="TX_LEAK_010",
            target_audience="analyst",
            fraud_probability=0.87,
            is_fraud_predicted=True,
            explanation_text=(
                "Transaction TX_LEAK_010 has a fraud probability of 87%. "
                "TransactionAmt SHAP contribution is 0.45."
            ),
            cited_features=["TransactionAmt"],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=False,
            explanation_generated=True,
            token_cost_usd=0.001,
            generation_latency_seconds=2.0,
        )
        assert "87%" in result.explanation_text

    def test_analyst_can_cite_shap_values(self):
        """Analyst mode is allowed to cite SHAP values directly."""
        result = ExplanationResult(
            transaction_id="TX_LEAK_011",
            target_audience="analyst",
            fraud_probability=0.92,
            is_fraud_predicted=True,
            explanation_text=(
                "SHAP analysis shows TransactionAmt (SHAP=0.55) is the primary driver. "
                "DeviceInfo (SHAP=0.30) is the second contributor."
            ),
            cited_features=["TransactionAmt", "DeviceInfo"],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=False,
            explanation_generated=True,
            token_cost_usd=0.001,
            generation_latency_seconds=2.0,
        )
        assert "SHAP" in result.explanation_text


# ---------------------------------------------------------------------------
# Hallucination = data fabrication leakage
# ---------------------------------------------------------------------------


class TestFeatureFabricationLeakage:
    """Citing features not in SHAP input is a form of data leakage:
    the LLM would be inventing data that doesn't exist."""

    def test_single_hallucinated_feature_rejected(self):
        with pytest.raises(ValidationError, match="ExplanationHallucinationError"):
            ExplanationResult(
                transaction_id="TX_LEAK_020",
                target_audience="analyst",
                fraud_probability=0.75,
                is_fraud_predicted=True,
                explanation_text="The IP address was flagged as suspicious.",
                cited_features=["ip_address"],
                uncited_features=[],
                hallucinated_features=["ip_address"],
                uncertainty_flag=False,
                explanation_generated=True,
                token_cost_usd=0.001,
                generation_latency_seconds=2.0,
            )

    def test_multiple_hallucinated_features_rejected(self):
        with pytest.raises(ValidationError, match="ExplanationHallucinationError"):
            ExplanationResult(
                transaction_id="TX_LEAK_021",
                target_audience="analyst",
                fraud_probability=0.80,
                is_fraud_predicted=True,
                explanation_text="IP and browser fingerprint were suspicious.",
                cited_features=["ip_address", "browser_fingerprint"],
                uncited_features=[],
                hallucinated_features=["ip_address", "browser_fingerprint"],
                uncertainty_flag=False,
                explanation_generated=True,
                token_cost_usd=0.001,
                generation_latency_seconds=2.0,
            )
