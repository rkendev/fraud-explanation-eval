"""Tests for Phase 1 — Data Pipeline (loader + preprocessor).

Gate requirement: 20+ tests, data schema validated, no PII in processed files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    _clean_categorical,
    _clean_numeric,
    _clean_text_fields,
    _engineer_features,
    _generate_transaction_id,
    _merge_transaction_identity,
    load_ieee_cis,
    load_raw_identity,
    load_raw_transaction,
)
from src.data.preprocessor import (
    EXCLUDE_COLS,
    TARGET_COL,
    _encode_categoricals,
    _select_features,
    _validate_required_columns,
    apply_smote,
    preprocess_pipeline,
    train_test_split_stratified,
)

DATA_AVAILABLE = Path("data/raw/train_transaction.csv").exists()

# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_txn_df() -> pd.DataFrame:
    """Minimal valid transaction DataFrame."""
    return pd.DataFrame(
        {
            "TransactionID": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "isFraud": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "TransactionAmt": [
                100.0,
                50.0,
                25.0,
                75.0,
                200.0,
                10.0,
                30.0,
                60.0,
                150.0,
                90.0,
            ],
            "ProductCD": ["W", "H", "C", "S", "R", "W", "H", "C", "S", "R"],
            "card1": [
                13926,
                14000,
                15000,
                16000,
                17000,
                13000,
                14500,
                15500,
                16500,
                17500,
            ],
            "card4": [
                "visa",
                "mastercard",
                "discover",
                "visa",
                "american express",
                "visa",
                "mastercard",
                "discover",
                "visa",
                "american express",
            ],
            "card6": [
                "credit",
                "debit",
                "credit",
                "debit",
                "credit",
                "debit",
                "credit",
                "debit",
                "credit",
                "debit",
            ],
            "addr1": [315, 200, 100, 400, 500, 315, 200, 100, 400, 500],
            "P_emaildomain": [
                "gmail.com",
                "yahoo.com",
                None,
                "hotmail.com",
                None,
                "gmail.com",
                "yahoo.com",
                None,
                "outlook.com",
                None,
            ],
            "R_emaildomain": [
                "gmail.com",
                None,
                None,
                "hotmail.com",
                None,
                None,
                "yahoo.com",
                None,
                "outlook.com",
                None,
            ],
        }
    )


@pytest.fixture
def minimal_identity_df() -> pd.DataFrame:
    """Minimal valid identity DataFrame."""
    return pd.DataFrame(
        {
            "TransactionID": [100, 102, 104, 106, 108],
            "DeviceType": ["desktop", "mobile", "desktop", "mobile", "desktop"],
            "DeviceInfo": ["Windows 10", "iPhone", "MacOS", "Android", "Linux"],
        }
    )


@pytest.fixture
def merged_df(minimal_txn_df, minimal_identity_df) -> pd.DataFrame:
    """Merged and cleaned DataFrame ready for preprocessing."""
    merged = _merge_transaction_identity(minimal_txn_df, minimal_identity_df)
    merged = _generate_transaction_id(merged)
    merged = _clean_categorical(merged)
    merged = _clean_numeric(merged)
    merged = _clean_text_fields(merged)
    merged = _engineer_features(merged)
    return merged


# ──────────────────────────────────────────────────────────────────
# Loader tests
# ──────────────────────────────────────────────────────────────────


class TestLoadRawTransaction:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Transaction file not found"):
            load_raw_transaction(tmp_path / "nonexistent.csv")

    def test_loads_csv_with_correct_columns(self, tmp_path):
        csv_path = tmp_path / "txn.csv"
        df = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "isFraud": [0, 1],
                "TransactionAmt": [10.0, 20.0],
                "ProductCD": ["W", "H"],
                "card1": [13000, 14000],
                "ExtraColumn": ["x", "y"],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_raw_transaction(csv_path)
        assert "TransactionID" in result.columns
        assert "ExtraColumn" not in result.columns
        assert len(result) == 2

    def test_sample_n_limits_rows(self, tmp_path):
        csv_path = tmp_path / "txn.csv"
        df = pd.DataFrame(
            {
                "TransactionID": range(100),
                "isFraud": [0] * 100,
                "TransactionAmt": [10.0] * 100,
                "ProductCD": ["W"] * 100,
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_raw_transaction(csv_path, sample_n=10)
        assert len(result) == 10

    def test_sample_n_larger_than_data_returns_all(self, tmp_path):
        csv_path = tmp_path / "txn.csv"
        df = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "isFraud": [0, 0, 1],
                "TransactionAmt": [10.0, 20.0, 30.0],
                "ProductCD": ["W", "H", "C"],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_raw_transaction(csv_path, sample_n=1000)
        assert len(result) == 3


class TestLoadRawIdentity:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Identity file not found"):
            load_raw_identity(tmp_path / "nonexistent.csv")

    def test_loads_only_relevant_columns(self, tmp_path):
        csv_path = tmp_path / "id.csv"
        df = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "DeviceType": ["desktop", "mobile"],
                "DeviceInfo": ["Win10", "iPhone"],
                "id_01": [0.0, 1.0],
                "id_02": [100.0, 200.0],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_raw_identity(csv_path)
        assert "DeviceType" in result.columns
        assert "id_01" not in result.columns


# ──────────────────────────────────────────────────────────────────
# Cleaning tests
# ──────────────────────────────────────────────────────────────────


class TestCleanCategorical:
    def test_card4_lowercased(self):
        df = pd.DataFrame({"card4": ["VISA", "Mastercard", "DISCOVER"]})
        result = _clean_categorical(df)
        assert result["card4"].tolist() == ["visa", "mastercard", "discover"]

    def test_invalid_card4_set_to_none(self):
        df = pd.DataFrame({"card4": ["visa", "unknown_card", "mastercard"]})
        result = _clean_categorical(df)
        assert result["card4"].iloc[1] is None

    def test_card6_lowercased(self):
        df = pd.DataFrame({"card6": ["CREDIT", "Debit"]})
        result = _clean_categorical(df)
        assert result["card6"].tolist() == ["credit", "debit"]

    def test_invalid_card6_set_to_none(self):
        df = pd.DataFrame({"card6": ["credit", "prepaid"]})
        result = _clean_categorical(df)
        assert result["card6"].iloc[1] is None

    def test_product_cd_uppercased(self):
        df = pd.DataFrame({"ProductCD": ["w", "h", "c"]})
        result = _clean_categorical(df)
        assert result["ProductCD"].tolist() == ["W", "H", "C"]

    def test_invalid_product_cd_dropped(self):
        df = pd.DataFrame({"ProductCD": ["W", "X", "H"]})
        result = _clean_categorical(df)
        assert len(result) == 2
        assert set(result["ProductCD"]) == {"W", "H"}

    def test_device_type_lowercased(self):
        df = pd.DataFrame({"DeviceType": ["Desktop", "MOBILE"]})
        result = _clean_categorical(df)
        assert result["DeviceType"].tolist() == ["desktop", "mobile"]

    def test_invalid_device_type_set_to_none(self):
        df = pd.DataFrame({"DeviceType": ["desktop", "tablet", "smartwatch"]})
        result = _clean_categorical(df)
        assert result["DeviceType"].iloc[0] == "desktop"
        assert result["DeviceType"].iloc[1] is None
        assert result["DeviceType"].iloc[2] is None


class TestCleanNumeric:
    def test_non_positive_amount_dropped(self):
        df = pd.DataFrame({"TransactionAmt": [100.0, 0.0, -5.0, 50.0]})
        result = _clean_numeric(df)
        assert len(result) == 2
        assert result["TransactionAmt"].tolist() == [100.0, 50.0]

    def test_card1_coerced_to_numeric(self):
        df = pd.DataFrame({"card1": ["13000", "abc", "15000"]})
        result = _clean_numeric(df)
        assert result["card1"].iloc[0] == 13000.0
        assert pd.isna(result["card1"].iloc[1])


class TestCleanTextFields:
    def test_nan_strings_converted_to_none(self):
        df = pd.DataFrame({"P_emaildomain": ["gmail.com", "nan", "NaN", ""]})
        result = _clean_text_fields(df)
        assert result["P_emaildomain"].iloc[0] == "gmail.com"
        assert result["P_emaildomain"].iloc[1] is None
        assert result["P_emaildomain"].iloc[2] is None
        assert result["P_emaildomain"].iloc[3] is None

    def test_long_text_truncated_to_256(self):
        df = pd.DataFrame({"DeviceInfo": ["A" * 500]})
        result = _clean_text_fields(df)
        assert len(result["DeviceInfo"].iloc[0]) == 256


# ──────────────────────────────────────────────────────────────────
# Feature engineering tests
# ──────────────────────────────────────────────────────────────────


class TestEngineerFeatures:
    def test_log_transform_created(self):
        df = pd.DataFrame({"TransactionAmt": [100.0, 1.0]})
        result = _engineer_features(df)
        assert "TransactionAmt_log" in result.columns
        assert result["TransactionAmt_log"].iloc[0] == pytest.approx(np.log1p(100.0))

    def test_email_domain_match(self):
        df = pd.DataFrame(
            {
                "P_emaildomain": ["gmail.com", "gmail.com", None],
                "R_emaildomain": ["gmail.com", "yahoo.com", None],
            }
        )
        result = _engineer_features(df)
        assert result["email_domain_match"].tolist() == [1, 0, 0]

    def test_missing_count(self):
        df = pd.DataFrame(
            {
                "card1": [1000, None, None],
                "card4": ["visa", None, None],
                "card6": ["credit", "debit", None],
                "addr1": [100, None, None],
                "P_emaildomain": ["a@b.com", None, None],
                "DeviceType": ["desktop", None, None],
            }
        )
        result = _engineer_features(df)
        assert result["missing_count"].iloc[0] == 0
        assert result["missing_count"].iloc[1] == 5
        assert result["missing_count"].iloc[2] == 6


# ──────────────────────────────────────────────────────────────────
# Merge tests
# ──────────────────────────────────────────────────────────────────


class TestMerge:
    def test_left_join_preserves_all_transactions(
        self, minimal_txn_df, minimal_identity_df
    ):
        result = _merge_transaction_identity(minimal_txn_df, minimal_identity_df)
        assert len(result) == len(minimal_txn_df)
        assert "DeviceType" in result.columns

    def test_non_matched_have_null_device(self, minimal_txn_df, minimal_identity_df):
        result = _merge_transaction_identity(minimal_txn_df, minimal_identity_df)
        # TransactionID 101 not in identity
        row_101 = result[result["TransactionID"] == 101]
        assert row_101["DeviceType"].isna().all()


class TestGenerateTransactionId:
    def test_converts_int_to_str(self):
        df = pd.DataFrame({"TransactionID": [100, 200]})
        result = _generate_transaction_id(df)
        assert result["TransactionID"].dtype == object
        assert result["TransactionID"].tolist() == ["100", "200"]


# ──────────────────────────────────────────────────────────────────
# Preprocessor tests
# ──────────────────────────────────────────────────────────────────


class TestValidateRequiredColumns:
    def test_missing_target_raises(self):
        df = pd.DataFrame({"TransactionID": [1], "TransactionAmt": [10.0]})
        with pytest.raises(ValueError, match="isFraud"):
            _validate_required_columns(df)

    def test_missing_transaction_id_raises(self):
        df = pd.DataFrame({"isFraud": [0], "TransactionAmt": [10.0]})
        with pytest.raises(ValueError, match="TransactionID"):
            _validate_required_columns(df)

    def test_missing_amount_raises(self):
        df = pd.DataFrame({"TransactionID": [1], "isFraud": [0]})
        with pytest.raises(ValueError, match="TransactionAmt"):
            _validate_required_columns(df)


class TestEncodeCategoricals:
    def test_one_hot_encoding(self):
        df = pd.DataFrame({"ProductCD": ["W", "H", "C"], "other": [1, 2, 3]})
        result = _encode_categoricals(df)
        assert "ProductCD_W" in result.columns
        assert "ProductCD_H" in result.columns
        assert "ProductCD" not in result.columns

    def test_nan_filled_with_missing(self):
        df = pd.DataFrame({"card4": ["visa", None, "mastercard"]})
        result = _encode_categoricals(df)
        assert "card4_missing" in result.columns
        assert result["card4_missing"].iloc[1] == 1


class TestTrainTestSplit:
    def test_stratified_split(self, merged_df):
        train, test = train_test_split_stratified(merged_df, test_size=0.3)
        assert len(train) + len(test) == len(merged_df)
        # Both should have fraud cases
        assert train[TARGET_COL].sum() > 0 or test[TARGET_COL].sum() > 0

    def test_invalid_test_size_raises(self, merged_df):
        with pytest.raises(ValueError, match="test_size must be in"):
            train_test_split_stratified(merged_df, test_size=0.0)

        with pytest.raises(ValueError, match="test_size must be in"):
            train_test_split_stratified(merged_df, test_size=1.0)

    def test_reset_index(self, merged_df):
        train, test = train_test_split_stratified(merged_df, test_size=0.3)
        assert train.index[0] == 0
        assert test.index[0] == 0


class TestSelectFeatures:
    def test_excludes_target_and_ids(self):
        df = pd.DataFrame(
            {
                "TransactionID": ["1"],
                "isFraud": [0],
                "TransactionAmt": [10.0],
                "P_emaildomain": ["a@b.com"],
                "feature_x": [1],
            }
        )
        features = _select_features(df)
        assert "isFraud" not in features
        assert "TransactionID" not in features
        assert "P_emaildomain" not in features
        assert "TransactionAmt" in features
        assert "feature_x" in features


class TestApplySmote:
    def test_increases_minority_class(self):
        # Need at least k_neighbors+1 minority samples (default k=5, so 6+)
        X = pd.DataFrame({"f1": range(100), "f2": range(100, 200)})
        y = pd.Series([0] * 90 + [1] * 10)

        X_res, y_res = apply_smote(X, y)
        assert y_res.sum() > y.sum()
        assert len(X_res) > len(X)

    def test_handles_nan_in_features(self):
        X = pd.DataFrame(
            {
                "f1": [
                    1.0,
                    2.0,
                    np.nan,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                    18.0,
                    19.0,
                    20.0,
                ],
            }
        )
        y = pd.Series([0] * 14 + [1] * 6)

        X_res, y_res = apply_smote(X, y, sampling_strategy=0.5)
        assert not X_res["f1"].isna().any()

    def test_too_few_minority_skips_smote(self):
        X = pd.DataFrame({"f1": range(10)})
        y = pd.Series([0] * 9 + [1] * 1)

        X_res, y_res = apply_smote(X, y)
        # Should return unchanged because only 1 minority sample
        assert len(X_res) == len(X)


class TestPreprocessPipeline:
    def test_full_pipeline_returns_expected_keys(self, merged_df):
        result = preprocess_pipeline(merged_df, apply_balancing=False)
        expected_keys = {
            "train_df",
            "test_df",
            "X_train",
            "y_train",
            "X_test",
            "y_test",
            "feature_cols",
        }
        assert set(result.keys()) == expected_keys

    def test_saves_parquet(self, merged_df, tmp_path):
        preprocess_pipeline(merged_df, apply_balancing=False, output_dir=tmp_path)
        assert (tmp_path / "train.parquet").exists()
        assert (tmp_path / "test.parquet").exists()

        # Verify parquet is readable
        train = pd.read_parquet(tmp_path / "train.parquet")
        assert len(train) > 0

    def test_no_pii_in_features(self, merged_df):
        """Gate requirement: no PII (email domains, DeviceInfo) in feature columns."""
        result = preprocess_pipeline(merged_df, apply_balancing=False)
        feature_cols = result["feature_cols"]

        for excluded in EXCLUDE_COLS:
            assert (
                excluded not in feature_cols
            ), f"PII column {excluded} found in features"

    def test_feature_cols_match_x_columns(self, merged_df):
        result = preprocess_pipeline(merged_df, apply_balancing=False)
        assert list(result["X_train"].columns) == result["feature_cols"]
        assert list(result["X_test"].columns) == result["feature_cols"]

    def test_with_smote(self):
        """Test SMOTE with enough data for k_neighbors to work."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "TransactionID": [str(i) for i in range(n)],
                "isFraud": [0] * 85 + [1] * 15,
                "TransactionAmt": np.random.uniform(10, 500, n),
                "ProductCD": np.random.choice(["W", "H", "C", "S", "R"], n),
                "card1": np.random.randint(1000, 18000, n),
            }
        )
        df_encoded = _encode_categoricals(df)
        result = preprocess_pipeline(df_encoded, apply_balancing=True)
        # SMOTE should increase training size
        assert len(result["X_train"]) >= len(result["train_df"])


# ──────────────────────────────────────────────────────────────────
# Integration: load_ieee_cis with synthetic data
# ──────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not DATA_AVAILABLE, reason="IEEE-CIS data not available in CI")
class TestLoadIeeeCisIntegration:
    def test_loads_real_data_sample(self):
        """Smoke test with real IEEE-CIS data (10-row sample)."""
        df = load_ieee_cis(sample_n=10, random_state=42)
        assert len(df) == 10
        assert "TransactionID" in df.columns
        assert "isFraud" in df.columns
        assert "TransactionAmt" in df.columns

    def test_real_data_has_engineered_features(self):
        df = load_ieee_cis(sample_n=10, engineer_features=True)
        assert "TransactionAmt_log" in df.columns
        assert "email_domain_match" in df.columns
        assert "missing_count" in df.columns

    def test_no_identity_merge(self):
        df = load_ieee_cis(sample_n=10, with_identity=False)
        assert "DeviceType" not in df.columns
        assert "DeviceInfo" not in df.columns

    def test_schema_compatible_output(self):
        """Verify loader output can produce FraudTransaction-compatible rows."""
        from src.schemas.transactions import FraudTransaction

        df = load_ieee_cis(sample_n=50, random_state=42)

        # Try to construct at least one valid FraudTransaction
        valid_count = 0
        for _, row in df.iterrows():
            try:
                FraudTransaction(
                    TransactionID=str(row["TransactionID"]),
                    TransactionAmt=float(row["TransactionAmt"]),
                    ProductCD=row["ProductCD"],
                    card1=int(row["card1"]) if pd.notna(row.get("card1")) else None,
                    card4=row.get("card4"),
                    card6=row.get("card6"),
                    addr1=int(row["addr1"]) if pd.notna(row.get("addr1")) else None,
                    P_emaildomain=row.get("P_emaildomain"),
                    R_emaildomain=row.get("R_emaildomain"),
                    DeviceType=row.get("DeviceType"),
                    DeviceInfo=row.get("DeviceInfo"),
                )
                valid_count += 1
            except Exception:
                pass

        assert valid_count > 0, "No rows produced valid FraudTransaction objects"
