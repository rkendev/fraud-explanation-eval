"""Preprocessing: train/test split, encoding, class balancing."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

TARGET_COL = "isFraud"

# Categorical columns to one-hot encode for XGBoost
CATEGORICAL_COLS = ["ProductCD", "card4", "card6", "DeviceType"]

# Numeric columns to keep as-is (NaN handled by XGBoost natively)
NUMERIC_COLS = [
    "TransactionAmt",
    "card1",
    "addr1",
    "TransactionAmt_log",
    "email_domain_match",
    "missing_count",
]

# Columns that are IDs or raw text — excluded from model features
EXCLUDE_COLS = [
    "TransactionID",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
]


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Check that the DataFrame has the minimum columns for training."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from DataFrame")
    if "TransactionID" not in df.columns:
        raise ValueError("TransactionID column missing from DataFrame")
    if "TransactionAmt" not in df.columns:
        raise ValueError("TransactionAmt column missing from DataFrame")


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns, dropping originals."""
    df = df.copy()
    present_cats = [c for c in CATEGORICAL_COLS if c in df.columns]

    if not present_cats:
        return df

    # Fill NaN with "missing" before encoding
    for col in present_cats:
        df[col] = df[col].fillna("missing")

    df = pd.get_dummies(df, columns=present_cats, prefix=present_cats, dtype=int)
    return df


def _select_features(df: pd.DataFrame) -> list[str]:
    """Return list of feature columns (everything except target and excluded)."""
    exclude = set(EXCLUDE_COLS) | {TARGET_COL}
    return [c for c in df.columns if c not in exclude]


def train_test_split_stratified(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split preserving fraud class ratio.

    Parameters
    ----------
    df : cleaned DataFrame with isFraud column
    test_size : fraction for test set
    random_state : reproducibility seed

    Returns
    -------
    (train_df, test_df) with same columns
    """
    _validate_required_columns(df)

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COL],
        random_state=random_state,
    )

    train_fraud_rate = train_df[TARGET_COL].mean()
    test_fraud_rate = test_df[TARGET_COL].mean()
    logger.info(
        "Split: train=%d (fraud=%.3f), test=%d (fraud=%.3f)",
        len(train_df),
        train_fraud_rate,
        len(test_df),
        test_fraud_rate,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    sampling_strategy: float = 0.5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance the training set.

    Parameters
    ----------
    X : feature matrix
    y : target series
    random_state : reproducibility seed
    sampling_strategy : ratio of minority to majority after resampling

    Returns
    -------
    (X_resampled, y_resampled)
    """
    from imblearn.over_sampling import SMOTE

    # SMOTE requires no NaN — fill numeric NaN with median
    X_filled = X.copy()
    for col in X_filled.columns:
        if X_filled[col].dtype in [
            np.float64,
            np.float32,
            np.int64,
            np.int32,
            float,
            int,
        ]:
            X_filled[col] = X_filled[col].fillna(X_filled[col].median())
        else:
            X_filled[col] = X_filled[col].fillna(0)

    minority_count = int(y.sum()) if y.mean() < 0.5 else int((1 - y).sum())
    k = min(5, minority_count - 1)
    if k < 1:
        logger.warning(
            "Too few minority samples (%d) for SMOTE, skipping", minority_count
        )
        return X_filled, y

    smote = SMOTE(
        random_state=random_state,
        sampling_strategy=sampling_strategy,
        k_neighbors=k,
    )
    X_resampled, y_resampled = smote.fit_resample(X_filled, y)

    logger.info(
        "SMOTE: %d -> %d rows (minority: %d -> %d)",
        len(X),
        len(X_resampled),
        y.sum(),
        y_resampled.sum(),
    )

    return pd.DataFrame(X_resampled, columns=X.columns), y_resampled


def preprocess_pipeline(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    apply_balancing: bool = True,
    random_state: int = 42,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame | pd.Series | list[str]]:
    """Full preprocessing pipeline: encode, split, balance, save.

    Parameters
    ----------
    df : cleaned DataFrame from loader
    test_size : fraction for test set
    apply_balancing : whether to apply SMOTE to training set
    random_state : reproducibility seed
    output_dir : if set, save train.parquet and test.parquet here

    Returns
    -------
    dict with keys:
        train_df, test_df : full DataFrames (with TransactionID, isFraud)
        X_train, y_train : training features and target (potentially SMOTE'd)
        X_test, y_test : test features and target
        feature_cols : list of feature column names
    """
    _validate_required_columns(df)

    # Encode categoricals
    df_encoded = _encode_categoricals(df)

    # Split
    train_df, test_df = train_test_split_stratified(
        df_encoded,
        test_size=test_size,
        random_state=random_state,
    )

    # Select features
    feature_cols = _select_features(train_df)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:10])

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    # Apply SMOTE to training set only
    if apply_balancing:
        X_train, y_train = apply_smote(X_train, y_train, random_state=random_state)

    # Save to parquet if output_dir specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_parquet(output_dir / "train.parquet", index=False)
        test_df.to_parquet(output_dir / "test.parquet", index=False)
        logger.info("Saved train.parquet and test.parquet to %s", output_dir)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }
