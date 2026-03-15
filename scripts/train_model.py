"""Train XGBoost model on IEEE-CIS data.

Usage: poetry run python scripts/train_model.py --sample 10000
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost fraud detector")
    parser.add_argument(
        "--sample", type=int, default=10000, help="Number of rows to sample"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw", help="Raw data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/artifacts",
        help="Model output directory",
    )
    args = parser.parse_args()

    from src.data.loader import load_ieee_cis
    from src.data.preprocessor import preprocess_pipeline
    from src.models.detector import FraudDetector
    from src.schemas.transactions import FraudTransaction

    # Load and preprocess data
    logger.info("Loading IEEE-CIS data (sample=%d)...", args.sample)
    df = load_ieee_cis(args.data_dir, sample_n=args.sample)
    logger.info("Loaded %d rows", len(df))

    result = preprocess_pipeline(df, apply_balancing=True, output_dir="data/processed")
    logger.info(
        "Preprocessed: %d train rows, %d test rows, %d features",
        len(result["X_train"]),
        len(result["X_test"]),
        len(result["feature_cols"]),
    )

    # Train
    detector = FraudDetector(
        model_path=f"{args.output_dir}/model.json",
        version_path=f"{args.output_dir}/version.txt",
    )
    metrics = detector.train(
        result["X_train"],
        result["y_train"],
        result["X_test"],
        result["y_test"],
    )
    logger.info("Training metrics: %s", metrics)

    # Save model
    detector.save()
    logger.info("Model saved to %s/model.json", args.output_dir)

    # Demo: predict on a sample transaction from test set
    test_df = result["test_df"]
    sample_row = test_df.iloc[0]

    # Reconstruct ProductCD from one-hot columns
    product_cd = "W"  # default
    for code in ["W", "H", "C", "S", "R"]:
        col_name = f"ProductCD_{code}"
        if col_name in sample_row.index and sample_row[col_name] == 1:
            product_cd = code
            break

    tx = FraudTransaction(
        TransactionID=str(sample_row.get("TransactionID", "SAMPLE_001")),
        TransactionAmt=float(sample_row["TransactionAmt"]),
        ProductCD=product_cd,
        card1=(
            int(sample_row["card1"]) if not _is_nan(sample_row.get("card1")) else None
        ),
        addr1=(
            int(sample_row["addr1"]) if not _is_nan(sample_row.get("addr1")) else None
        ),
    )

    feature_cols = result["feature_cols"]
    feature_row_data = {}
    for col in feature_cols:
        if col in sample_row.index:
            feature_row_data[col] = sample_row[col]
        else:
            feature_row_data[col] = 0

    import pandas as pd

    feature_row = pd.DataFrame([feature_row_data], columns=feature_cols)
    detection_result = detector.predict(tx, feature_row=feature_row)

    logger.info("=== Sample Prediction ===")
    logger.info("Transaction: %s", detection_result.transaction_id)
    logger.info("Fraud probability: %.4f", detection_result.fraud_probability)
    logger.info("Is fraud predicted: %s", detection_result.is_fraud_predicted)
    logger.info("Confidence tier: %s", detection_result.confidence_tier)
    logger.info("Inference latency: %.2f ms", detection_result.inference_latency_ms)
    logger.info("Top SHAP features:")
    for feat in detection_result.top_shap_features:
        logger.info(
            "  %s: shap=%.4f, value=%s",
            feat.feature_name,
            feat.shap_value,
            feat.feature_value,
        )
    logger.info("Model version: %s", detection_result.model_version)

    # Validate result against schema
    validated = detection_result.model_dump()
    logger.info(
        "FraudDetectionResult validated successfully: %d fields", len(validated)
    )


def _is_nan(val: object) -> bool:
    """Check if a value is NaN."""
    import math

    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False


if __name__ == "__main__":
    main()
