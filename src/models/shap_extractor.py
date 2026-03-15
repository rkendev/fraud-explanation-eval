"""SHAP feature extraction for XGBoost fraud detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.schemas.detection import SHAPFeature

logger = logging.getLogger(__name__)

# Maximum number of top features to return
TOP_K = 5


class SHAPComputationError(Exception):
    """Raised when SHAP value computation fails."""


class SHAPExtractor:
    """Extract top-K SHAP features from an XGBoost model prediction."""

    def __init__(self, model: Any, *, top_k: int = TOP_K) -> None:
        self._model = model
        self._top_k = top_k
        self._explainer: shap.TreeExplainer | None = None

    def _get_explainer(self) -> shap.TreeExplainer:
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        return self._explainer

    def extract(
        self,
        features: pd.DataFrame,
        feature_names: list[str],
    ) -> list[SHAPFeature]:
        """Compute SHAP values and return top-K features by |shap_value|.

        Parameters
        ----------
        features : single-row DataFrame of model features
        feature_names : ordered list of feature column names

        Returns
        -------
        list of SHAPFeature, sorted by |shap_value| descending, max top_k
        """
        try:
            explainer = self._get_explainer()
            shap_values = explainer.shap_values(features)
        except Exception as exc:
            raise SHAPComputationError(f"SHAP computation failed: {exc}") from exc

        # shap_values shape: (n_samples, n_features) for binary classification
        # We take the first (and only) row
        if isinstance(shap_values, list):
            # For older SHAP versions that return [class_0, class_1]
            values = np.array(shap_values[1][0])
        elif shap_values.ndim == 2:
            values = np.array(shap_values[0])
        else:
            values = np.array(shap_values)

        if len(values) != len(feature_names):
            raise SHAPComputationError(
                f"SHAP values length ({len(values)}) != "
                f"feature names length ({len(feature_names)})"
            )

        # Get top-K by absolute SHAP value
        abs_values = np.abs(values)
        top_indices = np.argsort(abs_values)[::-1][: self._top_k]

        row = features.iloc[0]
        result = []
        for idx in top_indices:
            feat_name = feature_names[idx]
            feat_val = row.iloc[idx]
            # Convert numpy types to native Python types
            if isinstance(feat_val, (np.integer,)):
                feat_val = int(feat_val)
            elif isinstance(feat_val, (np.floating,)):
                feat_val = float(feat_val)
            elif isinstance(feat_val, (np.bool_,)):
                feat_val = bool(feat_val)

            result.append(
                SHAPFeature(
                    feature_name=feat_name,
                    shap_value=float(values[idx]),
                    feature_value=feat_val,
                )
            )

        logger.debug(
            "SHAP top-%d features: %s",
            self._top_k,
            [(f.feature_name, f.shap_value) for f in result],
        )
        return result
