"""Adversarial hallucination tests for ExplanationAgent contracts.

These tests verify the schema-level enforcement of hallucination constraints.
They do NOT require a live LLM — they test the Pydantic validators directly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.explanation import ExplanationResult


def test_explanation_cannot_cite_feature_not_in_shap():
    """An explanation citing C1 when C1 was not in top_shap_features must fail."""
    with pytest.raises(ValidationError, match="ExplanationHallucinationError"):
        ExplanationResult(
            transaction_id="TX_ADV_001",
            target_audience="analyst",
            fraud_probability=0.82,
            is_fraud_predicted=True,
            explanation_text="The counter field C1 was anomalously high.",
            cited_features=["C1"],
            uncited_features=[],
            hallucinated_features=["C1"],  # validator should catch this
            uncertainty_flag=False,
            explanation_generated=True,
            token_cost_usd=0.001,
            generation_latency_seconds=2.0,
        )


def test_customer_explanation_cannot_state_probability_as_percent():
    """Customer explanation must not reveal '82%'."""
    with pytest.raises(ValidationError, match="must not contain fraud probability"):
        ExplanationResult(
            transaction_id="TX_ADV_002",
            target_audience="customer",
            fraud_probability=0.82,
            is_fraud_predicted=True,
            explanation_text="We detected an 82% chance your transaction is fraudulent.",
            cited_features=[],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=False,
            explanation_generated=True,
            token_cost_usd=0.001,
            generation_latency_seconds=2.0,
        )


def test_customer_explanation_cannot_state_raw_probability():
    """Customer explanation must not reveal '0.82'."""
    with pytest.raises(ValidationError, match="must not contain fraud probability"):
        ExplanationResult(
            transaction_id="TX_ADV_003",
            target_audience="customer",
            fraud_probability=0.82,
            is_fraud_predicted=True,
            explanation_text="The fraud score is 0.82 for this transaction.",
            cited_features=[],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=False,
            explanation_generated=True,
            token_cost_usd=0.001,
            generation_latency_seconds=2.0,
        )


def test_uncertain_explanation_must_disclose():
    """If uncertainty_flag=True, explanation must include disclosure text."""
    with pytest.raises(ValidationError, match="uncertainty_disclosure"):
        ExplanationResult(
            transaction_id="TX_ADV_004",
            target_audience="analyst",
            fraud_probability=0.51,
            is_fraud_predicted=True,
            explanation_text="Transaction was flagged.",
            cited_features=[],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=True,
            uncertainty_disclosure=None,  # missing!
            explanation_generated=True,
            token_cost_usd=0.001,
            generation_latency_seconds=2.0,
        )


def test_zero_cost_explanation_rejected():
    """token_cost_usd must never be 0.0 — enforces real LiteLLM integration."""
    with pytest.raises(ValidationError, match="must be populated from actual"):
        ExplanationResult(
            transaction_id="TX_ADV_005",
            target_audience="analyst",
            fraud_probability=0.82,
            is_fraud_predicted=True,
            explanation_text="Transaction flagged due to elevated TransactionAmt.",
            cited_features=["TransactionAmt"],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=False,
            explanation_generated=True,
            token_cost_usd=0.0,  # must be rejected
            generation_latency_seconds=2.0,
        )
