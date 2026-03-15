"""Tests for XGBoost fraud detector and SHAP extractor.

Phase 2 gate: 25+ tests, all failure modes, confidence tier boundaries,
FraudDetectionResult validated against contract.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.detector import (
    FraudDetector,
    InferenceTimeoutError,
    ModelNotLoadedError,
    _compute_confidence_tier,
    _load_model_version,
)
from src.models.shap_extractor import SHAPComputationError, SHAPExtractor
from src.schemas.detection import FRAUD_THRESHOLD, FraudDetectionResult, SHAPFeature
from src.schemas.transactions import FraudTransaction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_dataset():
    """Create a small reproducible dataset for training."""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame(
        {
            "TransactionAmt": rng.exponential(100, n),
            "card1": rng.randint(1000, 18000, n).astype(float),
            "addr1": rng.randint(100, 500, n).astype(float),
            "TransactionAmt_log": np.log1p(rng.exponential(100, n)),
            "email_domain_match": rng.randint(0, 2, n),
            "missing_count": rng.randint(0, 4, n),
            "ProductCD_W": rng.randint(0, 2, n),
            "ProductCD_H": rng.randint(0, 2, n),
            "card4_visa": rng.randint(0, 2, n),
            "card4_mastercard": rng.randint(0, 2, n),
            "card6_debit": rng.randint(0, 2, n),
            "card6_credit": rng.randint(0, 2, n),
            "DeviceType_desktop": rng.randint(0, 2, n),
            "DeviceType_mobile": rng.randint(0, 2, n),
        }
    )
    # Create target with ~20% fraud rate
    y = pd.Series((rng.random(n) < 0.2).astype(int))
    return X, y


# Small XGBoost params for fast tests
_TEST_XGB_PARAMS = {"n_estimators": 10, "max_depth": 3}


@pytest.fixture(scope="module")
def trained_detector(toy_dataset, tmp_path_factory):
    """Return a FraudDetector trained on the toy dataset, saved to tmp dir."""
    X, y = toy_dataset
    tmp_path = tmp_path_factory.mktemp("detector")
    model_path = tmp_path / "model.json"
    version_path = tmp_path / "version.txt"
    version_path.write_text("1.0.0\n")

    detector = FraudDetector(model_path=model_path, version_path=version_path)
    detector.train(X, y, xgb_params=_TEST_XGB_PARAMS)
    detector.save()
    return detector


@pytest.fixture
def sample_tx():
    """A valid FraudTransaction for inference."""
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


# ---------------------------------------------------------------------------
# TestComputeConfidenceTier
# ---------------------------------------------------------------------------


class TestComputeConfidenceTier:
    """Tests for _compute_confidence_tier boundary logic."""

    def test_high_confidence_above_08(self):
        assert _compute_confidence_tier(0.85) == "high"

    def test_high_confidence_below_02(self):
        assert _compute_confidence_tier(0.10) == "high"

    def test_medium_confidence_06_to_08(self):
        assert _compute_confidence_tier(0.70) == "medium"

    def test_medium_confidence_02_to_04(self):
        assert _compute_confidence_tier(0.30) == "medium"

    def test_low_confidence_04_to_06(self):
        assert _compute_confidence_tier(0.50) == "low"

    def test_boundary_08_is_medium(self):
        # 0.8 is in [0.6, 0.8] range -> medium
        assert _compute_confidence_tier(0.8) == "medium"

    def test_boundary_06_is_medium(self):
        assert _compute_confidence_tier(0.6) == "medium"

    def test_boundary_04_is_low(self):
        assert _compute_confidence_tier(0.4) == "low"

    def test_boundary_02_is_medium(self):
        assert _compute_confidence_tier(0.2) == "medium"

    def test_exact_zero_is_high(self):
        assert _compute_confidence_tier(0.0) == "high"

    def test_exact_one_is_high(self):
        # 1.0 > 0.8 -> high
        assert _compute_confidence_tier(1.0) == "high"


# ---------------------------------------------------------------------------
# TestModelVersionLoading
# ---------------------------------------------------------------------------


class TestModelVersionLoading:
    def test_load_version_from_file(self, tmp_path):
        vp = tmp_path / "version.txt"
        vp.write_text("2.3.1\n")
        assert _load_model_version(vp) == "2.3.1"

    def test_missing_version_file_raises(self, tmp_path):
        with pytest.raises(ModelNotLoadedError, match="Version file not found"):
            _load_model_version(tmp_path / "nonexistent.txt")


# ---------------------------------------------------------------------------
# TestFraudDetectorTraining
# ---------------------------------------------------------------------------


class TestFraudDetectorTraining:
    def test_train_returns_metrics(self, toy_dataset, tmp_path):
        X, y = toy_dataset
        version_path = tmp_path / "version.txt"
        version_path.write_text("1.0.0\n")

        detector = FraudDetector(
            model_path=tmp_path / "model.json", version_path=version_path
        )
        metrics = detector.train(X, y, xgb_params=_TEST_XGB_PARAMS)
        assert "train_auc" in metrics
        assert 0.0 <= metrics["train_auc"] <= 1.0

    def test_train_with_eval_set(self, toy_dataset, tmp_path):
        X, y = toy_dataset
        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        version_path = tmp_path / "version.txt"
        version_path.write_text("1.0.0\n")

        detector = FraudDetector(
            model_path=tmp_path / "model.json", version_path=version_path
        )
        metrics = detector.train(
            X_train, y_train, X_test, y_test, xgb_params=_TEST_XGB_PARAMS
        )
        assert "test_auc" in metrics
        assert 0.0 <= metrics["test_auc"] <= 1.0

    def test_model_is_loaded_after_training(self, trained_detector):
        assert trained_detector.is_loaded

    def test_feature_names_set_after_training(self, trained_detector):
        assert trained_detector.feature_names is not None
        assert len(trained_detector.feature_names) > 0


# ---------------------------------------------------------------------------
# TestFraudDetectorSaveLoad
# ---------------------------------------------------------------------------


class TestFraudDetectorSaveLoad:
    def test_save_creates_model_file(self, trained_detector):
        assert trained_detector._model_path.exists()

    def test_save_creates_feature_names_file(self, trained_detector):
        feat_path = trained_detector._model_path.with_suffix(".features.json")
        assert feat_path.exists()
        with open(feat_path) as f:
            names = json.load(f)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_load_restores_model(self, trained_detector):
        loaded = FraudDetector.load(
            model_path=trained_detector._model_path,
            version_path=trained_detector._version_path,
        )
        assert loaded.is_loaded
        assert loaded.model_version == "1.0.0"

    def test_load_missing_model_raises(self, tmp_path):
        with pytest.raises(ModelNotLoadedError, match="Model file not found"):
            FraudDetector.load(
                model_path=tmp_path / "nonexistent.json",
                version_path=tmp_path / "version.txt",
            )

    def test_save_without_model_raises(self, tmp_path):
        detector = FraudDetector(
            model_path=tmp_path / "model.json",
            version_path=tmp_path / "version.txt",
        )
        with pytest.raises(ModelNotLoadedError, match="No model to save"):
            detector.save()


# ---------------------------------------------------------------------------
# TestFraudDetectorInference
# ---------------------------------------------------------------------------


class TestFraudDetectorInference:
    def test_predict_returns_fraud_detection_result(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        assert isinstance(result, FraudDetectionResult)

    def test_predict_transaction_id_matches(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        assert result.transaction_id == "TX_TEST_001"

    def test_predict_probability_in_bounds(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        assert 0.0 <= result.fraud_probability <= 1.0

    def test_predict_is_fraud_consistent(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        expected = result.fraud_probability >= FRAUD_THRESHOLD
        assert result.is_fraud_predicted == expected

    def test_predict_confidence_tier_consistent(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        expected_tier = _compute_confidence_tier(result.fraud_probability)
        assert result.confidence_tier == expected_tier

    def test_predict_shap_features_returned(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        assert isinstance(result.top_shap_features, list)
        assert len(result.top_shap_features) <= 5

    def test_predict_shap_features_are_valid(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        for feat in result.top_shap_features:
            assert isinstance(feat, SHAPFeature)
            assert isinstance(feat.feature_name, str)
            assert isinstance(feat.shap_value, float)

    def test_predict_shap_sorted_by_abs_value(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        abs_values = [abs(f.shap_value) for f in result.top_shap_features]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_predict_model_version_set(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        assert result.model_version == "1.0.0"

    def test_predict_latency_recorded(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        assert result.inference_latency_ms > 0

    def test_predict_with_feature_row(self, trained_detector, sample_tx, toy_dataset):
        X, _ = toy_dataset
        feature_row = X.iloc[[0]]
        result = trained_detector.predict(sample_tx, feature_row=feature_row)
        assert isinstance(result, FraudDetectionResult)

    def test_predict_without_loaded_model_raises(self, tmp_path, sample_tx):
        version_path = tmp_path / "version.txt"
        version_path.write_text("1.0.0\n")
        detector = FraudDetector(
            model_path=tmp_path / "model.json", version_path=version_path
        )
        with pytest.raises(ModelNotLoadedError, match="Model not loaded"):
            detector.predict(sample_tx)

    def test_predict_with_minimal_transaction(self, trained_detector):
        tx = FraudTransaction(
            TransactionID="TX_MINIMAL",
            TransactionAmt=50.0,
            ProductCD="H",
        )
        result = trained_detector.predict(tx)
        assert isinstance(result, FraudDetectionResult)
        assert result.transaction_id == "TX_MINIMAL"

    def test_predict_batch(self, trained_detector, toy_dataset):
        X, _ = toy_dataset
        probas = trained_detector.predict_batch(X)
        assert len(probas) == len(X)
        assert all(0.0 <= p <= 1.0 for p in probas)

    def test_predict_batch_without_model_raises(self, tmp_path, toy_dataset):
        X, _ = toy_dataset
        detector = FraudDetector(
            model_path=tmp_path / "model.json",
            version_path=tmp_path / "version.txt",
        )
        with pytest.raises(ModelNotLoadedError, match="Model not loaded"):
            detector.predict_batch(X)


# ---------------------------------------------------------------------------
# TestFailureModes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_model_not_loaded_error(self, tmp_path, sample_tx):
        version_path = tmp_path / "version.txt"
        version_path.write_text("1.0.0\n")
        detector = FraudDetector(
            model_path=tmp_path / "model.json", version_path=version_path
        )
        with pytest.raises(ModelNotLoadedError):
            detector.predict(sample_tx)

    def test_inference_timeout_error(self, trained_detector, sample_tx):
        """Simulate timeout by setting a very low timeout threshold."""
        with patch("src.models.detector.INFERENCE_TIMEOUT_SECONDS", 0.0000001):
            with pytest.raises(InferenceTimeoutError):
                trained_detector.predict(sample_tx)

    def test_shap_computation_failure_returns_empty_features(
        self, trained_detector, sample_tx
    ):
        """When SHAP fails, result should have empty top_shap_features."""
        with patch.object(
            trained_detector._shap_extractor,
            "extract",
            side_effect=SHAPComputationError("mock failure"),
        ):
            result = trained_detector.predict(sample_tx)
            assert result.top_shap_features == []

    def test_feature_names_not_available_raises(self, tmp_path, sample_tx):
        version_path = tmp_path / "version.txt"
        version_path.write_text("1.0.0\n")
        detector = FraudDetector(
            model_path=tmp_path / "model.json", version_path=version_path
        )
        with pytest.raises(ModelNotLoadedError, match="no feature names"):
            detector.feature_names


# ---------------------------------------------------------------------------
# TestSHAPExtractor
# ---------------------------------------------------------------------------


class TestSHAPExtractor:
    def test_extract_returns_top_k_features(self, trained_detector, toy_dataset):
        X, _ = toy_dataset
        extractor = SHAPExtractor(trained_detector._model, top_k=5)
        features = extractor.extract(X.iloc[[0]], list(X.columns))
        assert len(features) == 5

    def test_extract_custom_top_k(self, trained_detector, toy_dataset):
        X, _ = toy_dataset
        extractor = SHAPExtractor(trained_detector._model, top_k=3)
        features = extractor.extract(X.iloc[[0]], list(X.columns))
        assert len(features) == 3

    def test_extract_sorted_by_abs_shap(self, trained_detector, toy_dataset):
        X, _ = toy_dataset
        extractor = SHAPExtractor(trained_detector._model, top_k=5)
        features = extractor.extract(X.iloc[[0]], list(X.columns))
        abs_vals = [abs(f.shap_value) for f in features]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_extract_feature_names_match_columns(self, trained_detector, toy_dataset):
        X, _ = toy_dataset
        extractor = SHAPExtractor(trained_detector._model, top_k=5)
        features = extractor.extract(X.iloc[[0]], list(X.columns))
        for feat in features:
            assert feat.feature_name in X.columns

    def test_extract_shap_computation_error(self):
        """SHAPExtractor raises SHAPComputationError on bad model."""
        mock_model = MagicMock()
        extractor = SHAPExtractor(mock_model, top_k=5)

        # Force the TreeExplainer to fail
        with patch(
            "src.models.shap_extractor.shap.TreeExplainer",
            side_effect=Exception("bad model"),
        ):
            with pytest.raises(SHAPComputationError, match="SHAP computation failed"):
                df = pd.DataFrame({"a": [1.0], "b": [2.0]})
                extractor.extract(df, ["a", "b"])

    def test_extract_handles_list_shap_values(self, trained_detector, toy_dataset):
        """Cover the branch for older SHAP returning [class_0, class_1] list."""
        X, _ = toy_dataset
        extractor = SHAPExtractor(trained_detector._model, top_k=3)
        features = X.iloc[[0]]
        feature_names = list(X.columns)

        # Mock shap_values to return a list of two arrays (old SHAP format)
        fake_values = np.random.RandomState(0).randn(len(feature_names))
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [
            np.array([fake_values * -1]),  # class 0
            np.array([fake_values]),  # class 1
        ]
        extractor._explainer = mock_explainer

        result = extractor.extract(features, feature_names)
        assert len(result) == 3

    def test_extract_handles_1d_shap_values(self, trained_detector, toy_dataset):
        """Cover the branch for 1D SHAP values (neither list nor 2D)."""
        X, _ = toy_dataset
        extractor = SHAPExtractor(trained_detector._model, top_k=3)
        features = X.iloc[[0]]
        feature_names = list(X.columns)

        fake_values = np.random.RandomState(0).randn(len(feature_names))
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = fake_values  # 1D array
        extractor._explainer = mock_explainer

        result = extractor.extract(features, feature_names)
        assert len(result) == 3

    def test_extract_length_mismatch_raises(self, trained_detector, toy_dataset):
        """Cover the branch for SHAP values length != feature names length."""
        X, _ = toy_dataset
        extractor = SHAPExtractor(trained_detector._model, top_k=3)
        features = X.iloc[[0]]

        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.array([[1.0, 2.0]])  # 2 values
        extractor._explainer = mock_explainer

        with pytest.raises(SHAPComputationError, match="SHAP values length"):
            extractor.extract(features, list(X.columns))  # 14 feature names

    def test_extract_converts_numpy_integer(self, trained_detector):
        """Cover np.integer conversion branch."""
        extractor = SHAPExtractor(trained_detector._model, top_k=2)

        # Build DataFrame with numpy integer columns
        df = pd.DataFrame(
            {"a": np.array([5], dtype=np.int64), "b": np.array([3], dtype=np.int64)}
        )
        fake_values = np.array([[0.5, -0.3]])
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = fake_values
        extractor._explainer = mock_explainer

        result = extractor.extract(df, ["a", "b"])
        assert isinstance(result[0].feature_value, int)

    def test_extract_converts_numpy_bool(self, trained_detector):
        """Cover np.bool_ conversion branch."""
        extractor = SHAPExtractor(trained_detector._model, top_k=2)

        df = pd.DataFrame(
            {
                "a": np.array([True], dtype=np.bool_),
                "b": np.array([False], dtype=np.bool_),
            }
        )
        fake_values = np.array([[0.5, -0.3]])
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = fake_values
        extractor._explainer = mock_explainer

        result = extractor.extract(df, ["a", "b"])
        assert isinstance(result[0].feature_value, bool)


# ---------------------------------------------------------------------------
# TestSchemaIntegration
# ---------------------------------------------------------------------------


class TestSchemaIntegration:
    """Verify that detector output validates against FraudDetectionResult schema."""

    def test_result_serializes_to_dict(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        d = result.model_dump()
        assert isinstance(d, dict)
        assert "transaction_id" in d
        assert "fraud_probability" in d
        assert "top_shap_features" in d

    def test_result_roundtrips_through_json(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        json_str = result.model_dump_json()
        restored = FraudDetectionResult.model_validate_json(json_str)
        assert restored.transaction_id == result.transaction_id
        assert restored.fraud_probability == result.fraud_probability

    def test_result_top_shap_max_five(self, trained_detector, sample_tx):
        result = trained_detector.predict(sample_tx)
        assert len(result.top_shap_features) <= 5
