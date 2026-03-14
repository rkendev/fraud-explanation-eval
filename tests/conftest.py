"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.schemas.detection import FraudDetectionResult, SHAPFeature
from src.schemas.explanation import ExplanationResult
from src.schemas.transactions import FraudTransaction


@pytest.fixture
def sample_transaction() -> FraudTransaction:
    return FraudTransaction(
        TransactionID="TX_TEST_001",
        TransactionAmt=299.99,
        ProductCD="W",
        card4="visa",
        card6="debit",
        addr1=325,
        P_emaildomain="gmail.com",
        R_emaildomain="gmail.com",
        DeviceType="desktop",
        DeviceInfo="Windows 10",
    )


@pytest.fixture
def high_fraud_detection_result() -> FraudDetectionResult:
    return FraudDetectionResult(
        transaction_id="TX_TEST_001",
        fraud_probability=0.87,
        is_fraud_predicted=True,
        top_shap_features=[
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
        ],
        model_version="1.0.0",
        inference_latency_ms=4.2,
        confidence_tier="high",
    )


@pytest.fixture
def low_confidence_detection_result() -> FraudDetectionResult:
    return FraudDetectionResult(
        transaction_id="TX_TEST_002",
        fraud_probability=0.52,
        is_fraud_predicted=True,
        top_shap_features=[
            SHAPFeature(
                feature_name="TransactionAmt", shap_value=0.08, feature_value=50.00
            ),
            SHAPFeature(feature_name="card4", shap_value=0.06, feature_value="visa"),
        ],
        model_version="1.0.0",
        inference_latency_ms=3.8,
        confidence_tier="low",
    )


@pytest.fixture
def valid_analyst_explanation(
    high_fraud_detection_result: FraudDetectionResult,
) -> ExplanationResult:
    return ExplanationResult(
        transaction_id="TX_TEST_001",
        target_audience="analyst",
        fraud_probability=0.87,
        is_fraud_predicted=True,
        explanation_text=(
            "Transaction TX_TEST_001 has been flagged with a fraud probability of 87%. "
            "The primary SHAP contributors are: TransactionAmt (0.45), "
            "DeviceInfo (0.31), and P_emaildomain (0.22). "
            "The transaction amount of $299.99 is consistent with elevated risk profiles."
        ),
        cited_features=["TransactionAmt", "DeviceInfo", "P_emaildomain"],
        uncited_features=["card6", "addr1"],
        hallucinated_features=[],
        uncertainty_flag=False,
        explanation_generated=True,
        token_cost_usd=0.001234,
        generation_latency_seconds=3.2,
    )


@pytest.fixture
def valid_customer_explanation() -> ExplanationResult:
    return ExplanationResult(
        transaction_id="TX_TEST_001",
        target_audience="customer",
        fraud_probability=0.87,
        is_fraud_predicted=True,
        explanation_text=(
            "Your recent transaction has been temporarily held for review. "
            "This is because we noticed unusual activity related to the transaction amount "
            "and the device used. If this was you, no action is needed."
        ),
        cited_features=["TransactionAmt", "DeviceInfo"],
        uncited_features=["P_emaildomain", "card6", "addr1"],
        hallucinated_features=[],
        uncertainty_flag=False,
        explanation_generated=True,
        token_cost_usd=0.000892,
        generation_latency_seconds=2.8,
    )
