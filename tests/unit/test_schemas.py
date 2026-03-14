"""Schema contract tests — all must pass before any agent is implemented."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.detection import FraudDetectionResult, SHAPFeature
from src.schemas.explanation import ExplanationEvalResult, ExplanationResult
from src.schemas.transactions import FraudTransaction

# ── FraudTransaction ──────────────────────────────────────────────────────────


def test_transaction_valid(sample_transaction: FraudTransaction):
    assert sample_transaction.TransactionAmt == 299.99
    assert sample_transaction.ProductCD == "W"


def test_transaction_negative_amount_rejected():
    with pytest.raises(ValidationError, match="must be positive"):
        FraudTransaction(TransactionID="TX1", TransactionAmt=-10.0, ProductCD="W")


def test_transaction_empty_id_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        FraudTransaction(TransactionID="   ", TransactionAmt=100.0, ProductCD="W")


def test_transaction_invalid_product_code_rejected():
    with pytest.raises(ValidationError):
        FraudTransaction(TransactionID="TX1", TransactionAmt=100.0, ProductCD="Z")


def test_transaction_long_device_info_truncated():
    long_info = "A" * 500
    tx = FraudTransaction(
        TransactionID="TX1", TransactionAmt=100.0, ProductCD="W", DeviceInfo=long_info
    )
    assert len(tx.DeviceInfo) == 256


# ── FraudDetectionResult ──────────────────────────────────────────────────────


def test_detection_result_valid(high_fraud_detection_result: FraudDetectionResult):
    assert high_fraud_detection_result.fraud_probability == 0.87
    assert high_fraud_detection_result.is_fraud_predicted is True
    assert high_fraud_detection_result.confidence_tier == "high"
    assert len(high_fraud_detection_result.top_shap_features) == 5


def test_detection_probability_out_of_bounds():
    with pytest.raises(ValidationError, match="fraud_probability must be in"):
        FraudDetectionResult(
            transaction_id="TX1",
            fraud_probability=1.5,
            is_fraud_predicted=True,
            top_shap_features=[],
            model_version="1.0.0",
            inference_latency_ms=1.0,
            confidence_tier="high",
        )


def test_detection_confidence_tier_inconsistent_with_probability():
    with pytest.raises(ValidationError, match="inconsistent"):
        FraudDetectionResult(
            transaction_id="TX1",
            fraud_probability=0.87,
            is_fraud_predicted=True,
            top_shap_features=[],
            model_version="1.0.0",
            inference_latency_ms=1.0,
            confidence_tier="low",  # wrong tier
        )


def test_detection_is_fraud_inconsistent_with_probability():
    with pytest.raises(ValidationError, match="inconsistent"):
        FraudDetectionResult(
            transaction_id="TX1",
            fraud_probability=0.87,
            is_fraud_predicted=False,  # wrong
            top_shap_features=[],
            model_version="1.0.0",
            inference_latency_ms=1.0,
            confidence_tier="high",
        )


def test_detection_too_many_shap_features():
    features = [
        SHAPFeature(feature_name=f"f{i}", shap_value=0.1, feature_value=i)
        for i in range(6)
    ]
    with pytest.raises(ValidationError, match="≤5 entries"):
        FraudDetectionResult(
            transaction_id="TX1",
            fraud_probability=0.87,
            is_fraud_predicted=True,
            top_shap_features=features,
            model_version="1.0.0",
            inference_latency_ms=1.0,
            confidence_tier="high",
        )


def test_confidence_tier_low_for_uncertain_probability():
    result = FraudDetectionResult(
        transaction_id="TX1",
        fraud_probability=0.52,
        is_fraud_predicted=True,
        top_shap_features=[],
        model_version="1.0.0",
        inference_latency_ms=1.0,
        confidence_tier="low",
    )
    assert result.confidence_tier == "low"


# ── ExplanationResult ─────────────────────────────────────────────────────────


def test_explanation_analyst_valid(valid_analyst_explanation: ExplanationResult):
    assert valid_analyst_explanation.target_audience == "analyst"
    assert valid_analyst_explanation.hallucinated_features == []
    assert valid_analyst_explanation.token_cost_usd > 0.0


def test_explanation_customer_valid(valid_customer_explanation: ExplanationResult):
    assert valid_customer_explanation.target_audience == "customer"
    assert "87%" not in valid_customer_explanation.explanation_text
    assert "0.87" not in valid_customer_explanation.explanation_text


def test_explanation_hallucinated_feature_rejected():
    with pytest.raises(ValidationError, match="ExplanationHallucinationError"):
        ExplanationResult(
            transaction_id="TX1",
            target_audience="analyst",
            fraud_probability=0.87,
            is_fraud_predicted=True,
            explanation_text="The C1 counter field was elevated.",
            cited_features=["C1"],  # C1 was NOT in top_shap_features
            uncited_features=[],
            hallucinated_features=["C1"],  # correctly detected as hallucination
            uncertainty_flag=False,
            explanation_generated=True,
            token_cost_usd=0.001,
            generation_latency_seconds=3.0,
        )


def test_explanation_customer_probability_leakage_rejected():
    with pytest.raises(ValidationError, match="must not contain fraud probability"):
        ExplanationResult(
            transaction_id="TX1",
            target_audience="customer",
            fraud_probability=0.87,
            is_fraud_predicted=True,
            explanation_text="Your transaction has an 87% fraud probability.",  # leaks!
            cited_features=[],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=False,
            explanation_generated=True,
            token_cost_usd=0.001,
            generation_latency_seconds=3.0,
        )


def test_explanation_uncertainty_not_disclosed_rejected():
    with pytest.raises(ValidationError, match="uncertainty_disclosure"):
        ExplanationResult(
            transaction_id="TX1",
            target_audience="analyst",
            fraud_probability=0.52,
            is_fraud_predicted=True,
            explanation_text="Transaction flagged.",
            cited_features=[],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=True,  # flag set
            uncertainty_disclosure=None,  # but not disclosed!
            explanation_generated=True,
            token_cost_usd=0.001,
            generation_latency_seconds=3.0,
        )


def test_explanation_zero_cost_rejected():
    with pytest.raises(ValidationError, match="must be populated from actual"):
        ExplanationResult(
            transaction_id="TX1",
            target_audience="analyst",
            fraud_probability=0.87,
            is_fraud_predicted=True,
            explanation_text="Transaction flagged due to TransactionAmt elevation.",
            cited_features=["TransactionAmt"],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=False,
            explanation_generated=True,
            token_cost_usd=0.0,  # rejected — must be real
            generation_latency_seconds=3.0,
        )


def test_explanation_word_limit_enforced():
    long_text = "word " * 350
    with pytest.raises(ValidationError, match="exceeds 300 words"):
        ExplanationResult(
            transaction_id="TX1",
            target_audience="analyst",
            fraud_probability=0.87,
            is_fraud_predicted=True,
            explanation_text=long_text,
            cited_features=[],
            uncited_features=[],
            hallucinated_features=[],
            uncertainty_flag=False,
            explanation_generated=True,
            token_cost_usd=0.001,
            generation_latency_seconds=3.0,
        )


# ── ExplanationEvalResult ─────────────────────────────────────────────────────


def test_eval_result_passed_consistent_with_score():
    result = ExplanationEvalResult(
        transaction_id="TX1",
        target_audience="analyst",
        grounding_score=0.9,
        clarity_score=0.85,
        completeness_score=0.8,
        audience_appropriateness_score=0.9,
        overall_score=0.86,
        passed=True,
        failure_reasons=[],
        token_cost_usd=0.015,
    )
    assert result.passed is True


def test_eval_result_failed_consistent_with_score():
    with pytest.raises(ValidationError, match="inconsistent"):
        ExplanationEvalResult(
            transaction_id="TX1",
            target_audience="analyst",
            grounding_score=0.4,
            clarity_score=0.5,
            completeness_score=0.3,
            audience_appropriateness_score=0.4,
            overall_score=0.40,
            passed=True,  # wrong — below 0.70 threshold
            failure_reasons=[],
            token_cost_usd=0.015,
        )
