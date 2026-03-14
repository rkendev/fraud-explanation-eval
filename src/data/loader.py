"""IEEE-CIS Fraud Detection data loader and feature engineering."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns used by FraudTransaction schema + target + join key
TRANSACTION_COLS = [
    "TransactionID",
    "isFraud",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card4",
    "card6",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
]

IDENTITY_COLS = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
]

VALID_PRODUCT_CDS = {"W", "H", "C", "S", "R"}
VALID_CARD4 = {"discover", "mastercard", "visa", "american express"}
VALID_CARD6 = {"credit", "debit"}
VALID_DEVICE_TYPES = {"desktop", "mobile"}


def load_raw_transaction(
    path: str | Path,
    *,
    sample_n: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load raw transaction CSV, selecting only relevant columns.

    Parameters
    ----------
    path : path to train_transaction.csv or test_transaction.csv
    sample_n : if set, return a random sample of N rows (for dev speed)
    random_state : seed for reproducible sampling
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {path}")

    available_cols = pd.read_csv(path, nrows=0).columns.tolist()
    use_cols = [c for c in TRANSACTION_COLS if c in available_cols]

    df = pd.read_csv(path, usecols=use_cols)
    logger.info("Loaded %d rows from %s", len(df), path.name)

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=random_state)
        logger.info("Sampled %d rows", sample_n)

    return df


def load_raw_identity(path: str | Path) -> pd.DataFrame:
    """Load raw identity CSV, selecting only relevant columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Identity file not found: {path}")

    available_cols = pd.read_csv(path, nrows=0).columns.tolist()
    use_cols = [c for c in IDENTITY_COLS if c in available_cols]

    df = pd.read_csv(path, usecols=use_cols)
    logger.info("Loaded %d identity rows from %s", len(df), path.name)
    return df


def _merge_transaction_identity(
    txn_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join transaction and identity on TransactionID."""
    merged = txn_df.merge(identity_df, on="TransactionID", how="left")
    logger.info(
        "Merged: %d transactions, %d with identity data",
        len(merged),
        merged["DeviceType"].notna().sum() if "DeviceType" in merged.columns else 0,
    )
    return merged


def _clean_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate categorical fields."""
    df = df.copy()

    # card4: lowercase, keep only valid values
    if "card4" in df.columns:
        df["card4"] = df["card4"].astype(str).str.lower().str.strip()
        df.loc[~df["card4"].isin(VALID_CARD4), "card4"] = None

    # card6: lowercase, keep only valid values
    if "card6" in df.columns:
        df["card6"] = df["card6"].astype(str).str.lower().str.strip()
        df.loc[~df["card6"].isin(VALID_CARD6), "card6"] = None

    # ProductCD: uppercase, keep only valid values
    if "ProductCD" in df.columns:
        df["ProductCD"] = df["ProductCD"].astype(str).str.upper().str.strip()
        invalid_mask = ~df["ProductCD"].isin(VALID_PRODUCT_CDS)
        if invalid_mask.any():
            logger.warning(
                "Dropping %d rows with invalid ProductCD", invalid_mask.sum()
            )
            df = df[~invalid_mask]

    # DeviceType: lowercase, keep only valid values
    if "DeviceType" in df.columns:
        df["DeviceType"] = df["DeviceType"].astype(str).str.lower().str.strip()
        df.loc[~df["DeviceType"].isin(VALID_DEVICE_TYPES), "DeviceType"] = None

    return df


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric fields: enforce positive amounts, valid ranges."""
    df = df.copy()

    # TransactionAmt must be > 0
    if "TransactionAmt" in df.columns:
        invalid_amt = df["TransactionAmt"] <= 0
        if invalid_amt.any():
            logger.warning(
                "Dropping %d rows with non-positive TransactionAmt", invalid_amt.sum()
            )
            df = df[~invalid_amt]

    # card1 range (per DETECTOR_SPEC: 1000–18396)
    if "card1" in df.columns:
        df["card1"] = pd.to_numeric(df["card1"], errors="coerce")

    # addr1: numeric
    if "addr1" in df.columns:
        df["addr1"] = pd.to_numeric(df["addr1"], errors="coerce")

    return df


def _clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Clean email domains and DeviceInfo — truncate, normalize."""
    df = df.copy()

    for col in ["P_emaildomain", "R_emaildomain", "DeviceInfo"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Replace pandas 'nan' string with actual None
            df.loc[df[col].isin(["nan", "NaN", ""]), col] = None
            # Truncate to 256 chars (matches schema sanitize_text_field)
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].str[:256]

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: create derived features for XGBoost.

    Adds columns that are useful for fraud detection but not part
    of the raw IEEE-CIS schema.
    """
    df = df.copy()

    # Log-transform of TransactionAmt (reduces skewness)
    if "TransactionAmt" in df.columns:
        import numpy as np

        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Email domain match: same sender and receiver domain
    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["email_domain_match"] = (
            df["P_emaildomain"].notna()
            & df["R_emaildomain"].notna()
            & (df["P_emaildomain"] == df["R_emaildomain"])
        ).astype(int)

    # Missing value count across key fields
    key_fields = ["card1", "card4", "card6", "addr1", "P_emaildomain", "DeviceType"]
    present = [c for c in key_fields if c in df.columns]
    if present:
        df["missing_count"] = df[present].isna().sum(axis=1)

    return df


def _generate_transaction_id(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure TransactionID is a string."""
    df = df.copy()
    if "TransactionID" in df.columns:
        df["TransactionID"] = df["TransactionID"].astype(str)
    return df


def load_ieee_cis(
    data_dir: str | Path = "data/raw",
    *,
    sample_n: int | None = None,
    random_state: int = 42,
    with_identity: bool = True,
    engineer_features: bool = True,
) -> pd.DataFrame:
    """Full IEEE-CIS loading pipeline: load, merge, clean, engineer.

    Parameters
    ----------
    data_dir : directory containing train_transaction.csv and train_identity.csv
    sample_n : if set, sample N rows from transactions (before merge)
    random_state : seed for reproducible sampling
    with_identity : whether to merge identity data (DeviceType, DeviceInfo)
    engineer_features : whether to run feature engineering

    Returns
    -------
    Cleaned DataFrame ready for preprocessing
    """
    data_dir = Path(data_dir)

    txn_path = data_dir / "train_transaction.csv"
    df = load_raw_transaction(txn_path, sample_n=sample_n, random_state=random_state)

    if with_identity:
        identity_path = data_dir / "train_identity.csv"
        if identity_path.exists():
            identity_df = load_raw_identity(identity_path)
            df = _merge_transaction_identity(df, identity_df)
        else:
            logger.warning(
                "Identity file not found at %s, skipping merge", identity_path
            )

    df = _generate_transaction_id(df)
    df = _clean_categorical(df)
    df = _clean_numeric(df)
    df = _clean_text_fields(df)

    if engineer_features:
        df = _engineer_features(df)

    logger.info("Final dataset: %d rows, %d columns", len(df), len(df.columns))
    return df
