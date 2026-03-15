"""XGBoost fraud detector: training, saving, loading, and inference."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.shap_extractor import SHAPComputationError, SHAPExtractor
from src.schemas.detection import FRAUD_THRESHOLD, FraudDetectionResult, SHAPFeature
from src.schemas.transactions import FraudTransaction

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MODEL_PATH = Path("models/artifacts/model.json")
DEFAULT_VERSION_PATH = Path("models/artifacts/version.txt")

# Inference timeout in seconds (spec: 2 seconds)
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "2.0"))


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted without a loaded model."""


class TransactionValidationError(ValueError):
    """Raised when a transaction fails feature validation."""


class InferenceTimeoutError(TimeoutError):
    """Raised when inference exceeds the allowed time limit."""


# XGBoost hyperparameters for fraud detection
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 1,  # Adjusted during training based on class imbalance
    "random_state": 42,
    "n_jobs": -1,
}

# Feature columns the detector expects (from preprocessor)
# These are the raw column names before one-hot encoding
RAW_FEATURE_COLS = [
    "TransactionAmt",
    "card1",
    "addr1",
    "TransactionAmt_log",
    "email_domain_match",
    "missing_count",
]

CATEGORICAL_COLS = ["ProductCD", "card4", "card6", "DeviceType"]


def _compute_confidence_tier(probability: float) -> str:
    """Map fraud probability to confidence tier per DETECTOR_SPEC."""
    if probability > 0.8 or probability < 0.2:
        return "high"
    elif (0.6 <= probability <= 0.8) or (0.2 <= probability < 0.4):
        return "medium"
    else:
        return "low"


def _load_model_version(version_path: Path) -> str:
    """Load model version string from version.txt."""
    if not version_path.exists():
        raise ModelNotLoadedError(f"Version file not found: {version_path}")
    return version_path.read_text().strip()


class FraudDetector:
    """XGBoost-based fraud detector with SHAP explanations.

    Usage
    -----
    # Training
    detector = FraudDetector()
    detector.train(X_train, y_train, X_test, y_test)
    detector.save()

    # Inference
    detector = FraudDetector.load()
    result = detector.predict(transaction, feature_row)
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        version_path: str | Path = DEFAULT_VERSION_PATH,
    ) -> None:
        self._model_path = Path(model_path)
        self._version_path = Path(version_path)
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] | None = None
        self._shap_extractor: SHAPExtractor | None = None
        self._model_version: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> str:
        if self._model_version is None:
            self._model_version = _load_model_version(self._version_path)
        return self._model_version

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise ModelNotLoadedError("Model not loaded — no feature names available")
        return self._feature_names

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        *,
        xgb_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train the XGBoost model.

        Parameters
        ----------
        X_train : training feature matrix
        y_train : training labels (0/1)
        X_test : optional test features for evaluation
        y_test : optional test labels for evaluation
        xgb_params : override default hyperparameters

        Returns
        -------
        dict with training metrics: train_auc, test_auc (if test provided)
        """
        params = {**DEFAULT_XGB_PARAMS}
        if xgb_params:
            params.update(xgb_params)

        # Compute scale_pos_weight from class imbalance
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        if pos_count > 0:
            params["scale_pos_weight"] = neg_count / pos_count
            logger.info(
                "Class balance: neg=%d, pos=%d, scale_pos_weight=%.2f",
                neg_count,
                pos_count,
                params["scale_pos_weight"],
            )

        self._feature_names = list(X_train.columns)
        self._model = xgb.XGBClassifier(**params)

        eval_set = [(X_train, y_train)]
        if X_test is not None and y_test is not None:
            eval_set.append((X_test, y_test))

        self._model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Reinitialize SHAP extractor with new model
        self._shap_extractor = SHAPExtractor(self._model)

        metrics: dict[str, Any] = {}
        from sklearn.metrics import roc_auc_score

        train_pred = self._model.predict_proba(X_train)[:, 1]
        metrics["train_auc"] = roc_auc_score(y_train, train_pred)
        logger.info("Train AUC: %.4f", metrics["train_auc"])

        if X_test is not None and y_test is not None:
            test_pred = self._model.predict_proba(X_test)[:, 1]
            metrics["test_auc"] = roc_auc_score(y_test, test_pred)
            logger.info("Test AUC: %.4f", metrics["test_auc"])

        return metrics

    def save(
        self,
        model_path: str | Path | None = None,
        version_path: str | Path | None = None,
        feature_names_path: str | Path | None = None,
    ) -> None:
        """Save model, version, and feature names to disk."""
        if self._model is None:
            raise ModelNotLoadedError("No model to save — train first")

        model_path = Path(model_path or self._model_path)
        version_path = Path(version_path or self._version_path)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.get_booster().save_model(str(model_path))
        logger.info("Model saved to %s", model_path)

        # Save feature names alongside model
        feat_path = feature_names_path or model_path.with_suffix(".features.json")
        import json

        with open(feat_path, "w") as f:
            json.dump(self._feature_names, f)
        logger.info("Feature names saved to %s", feat_path)

    @classmethod
    def load(
        cls,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        version_path: str | Path = DEFAULT_VERSION_PATH,
    ) -> FraudDetector:
        """Load a trained model from disk."""
        model_path = Path(model_path)
        version_path = Path(version_path)

        if not model_path.exists():
            raise ModelNotLoadedError(f"Model file not found: {model_path}")

        detector = cls(model_path=model_path, version_path=version_path)
        detector._model = xgb.XGBClassifier()
        detector._model.load_model(str(model_path))
        detector._model_version = _load_model_version(version_path)
        detector._shap_extractor = SHAPExtractor(detector._model)

        # Load feature names
        import json

        feat_path = model_path.with_suffix(".features.json")
        if feat_path.exists():
            with open(feat_path) as f:
                detector._feature_names = json.load(f)
            logger.info(
                "Loaded model v%s with %d features from %s",
                detector._model_version,
                len(detector._feature_names),
                model_path,
            )
        else:
            logger.warning("Feature names file not found: %s", feat_path)
            detector._feature_names = None

        return detector

    def _prepare_features(self, transaction: FraudTransaction) -> pd.DataFrame:
        """Convert a FraudTransaction to a single-row feature DataFrame.

        Applies the same encoding and feature engineering as the
        preprocessor pipeline, but for a single transaction.
        """
        if self._feature_names is None:
            raise ModelNotLoadedError(
                "Feature names not available — model may not be properly loaded"
            )

        data: dict[str, Any] = {}

        # Numeric features
        data["TransactionAmt"] = transaction.TransactionAmt
        data["card1"] = (
            float(transaction.card1) if transaction.card1 is not None else np.nan
        )
        data["addr1"] = (
            float(transaction.addr1) if transaction.addr1 is not None else np.nan
        )

        # Engineered features
        data["TransactionAmt_log"] = np.log1p(transaction.TransactionAmt)

        # Email domain match
        if (
            transaction.P_emaildomain is not None
            and transaction.R_emaildomain is not None
        ):
            data["email_domain_match"] = int(
                transaction.P_emaildomain == transaction.R_emaildomain
            )
        else:
            data["email_domain_match"] = 0

        # Missing count
        key_fields = [
            transaction.card1,
            transaction.card4,
            transaction.card6,
            transaction.addr1,
            transaction.P_emaildomain,
            transaction.DeviceType,
        ]
        data["missing_count"] = sum(1 for f in key_fields if f is None)

        # One-hot encode categoricals
        for cat_col in CATEGORICAL_COLS:
            val = getattr(transaction, cat_col, None)
            if val is None:
                val = "missing"
            # Create one-hot columns matching the preprocessor pattern
            for feat in self._feature_names:
                if feat.startswith(f"{cat_col}_"):
                    suffix = feat[len(cat_col) + 1 :]
                    data[feat] = 1 if suffix == val else 0

        # Build DataFrame with exact feature order
        row = {}
        for feat in self._feature_names:
            row[feat] = data.get(feat, 0)

        df = pd.DataFrame([row], columns=self._feature_names)
        return df

    def predict(
        self,
        transaction: FraudTransaction,
        feature_row: pd.DataFrame | None = None,
    ) -> FraudDetectionResult:
        """Run inference on a single transaction.

        Parameters
        ----------
        transaction : validated FraudTransaction
        feature_row : optional pre-prepared feature DataFrame (single row).
                      If None, features are prepared from the transaction.

        Returns
        -------
        FraudDetectionResult with fraud probability, SHAP features, etc.
        """
        if self._model is None:
            raise ModelNotLoadedError("Model not loaded — call load() or train() first")

        start_time = time.perf_counter()

        # Prepare features
        if feature_row is None:
            features = self._prepare_features(transaction)
        else:
            features = feature_row

        feature_names = list(features.columns)

        # Get probability
        proba = self._model.predict_proba(features)[:, 1][0]

        elapsed = time.perf_counter() - start_time
        if elapsed > INFERENCE_TIMEOUT_SECONDS:
            raise InferenceTimeoutError(
                f"Inference took {elapsed:.3f}s, exceeds {INFERENCE_TIMEOUT_SECONDS}s limit"
            )

        fraud_probability = float(proba)
        is_fraud = fraud_probability >= FRAUD_THRESHOLD

        # Extract SHAP features
        shap_features: list[SHAPFeature] = []
        try:
            if self._shap_extractor is None:
                self._shap_extractor = SHAPExtractor(self._model)
            shap_features = self._shap_extractor.extract(features, feature_names)
        except SHAPComputationError as exc:
            logger.warning("SHAP computation failed: %s", exc)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if elapsed_ms / 1000 > INFERENCE_TIMEOUT_SECONDS:
            raise InferenceTimeoutError(
                f"Inference took {elapsed_ms:.1f}ms, exceeds "
                f"{INFERENCE_TIMEOUT_SECONDS * 1000:.0f}ms limit"
            )

        confidence_tier = _compute_confidence_tier(fraud_probability)

        return FraudDetectionResult(
            transaction_id=transaction.TransactionID,
            fraud_probability=fraud_probability,
            is_fraud_predicted=is_fraud,
            top_shap_features=shap_features,
            model_version=self.model_version,
            inference_latency_ms=elapsed_ms,
            confidence_tier=confidence_tier,
        )

    def predict_batch(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Batch prediction — returns array of fraud probabilities.

        For use in training evaluation, not for producing FraudDetectionResult.
        """
        if self._model is None:
            raise ModelNotLoadedError("Model not loaded")
        return self._model.predict_proba(X)[:, 1]
