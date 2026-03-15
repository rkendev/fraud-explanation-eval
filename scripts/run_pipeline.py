"""Run full pipeline for a single transaction.

Usage: poetry run python scripts/run_pipeline.py --tx TX_TEST_001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path for `src.*` imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv

load_dotenv()

# Sample transactions for CLI use and testing
SAMPLE_TRANSACTIONS: dict[str, dict] = {
    "TX_TEST_001": {
        "TransactionID": "TX_TEST_001",
        "TransactionAmt": 299.99,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit",
        "addr1": 325,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "DeviceInfo": "Windows 10",
    },
    "TX_TEST_002": {
        "TransactionID": "TX_TEST_002",
        "TransactionAmt": 50.00,
        "ProductCD": "H",
        "card4": "mastercard",
        "card6": "credit",
        "addr1": 100,
        "P_emaildomain": "yahoo.com",
        "DeviceType": "mobile",
        "DeviceInfo": "iPhone",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fraud explanation pipeline")
    parser.add_argument("--tx", required=True, help="Transaction ID (or a known sample ID)")
    parser.add_argument(
        "--audience",
        choices=["analyst", "customer"],
        default="analyst",
        help="Target audience for explanation",
    )
    args = parser.parse_args()

    from src.models.detector import FraudDetector
    from src.orchestrator.graph import run_pipeline
    from src.schemas.transactions import FraudTransaction

    # Build transaction
    tx_data = SAMPLE_TRANSACTIONS.get(args.tx)
    if tx_data is None:
        print(f"[error] Unknown transaction ID: {args.tx}")
        print(f"[info]  Available samples: {', '.join(SAMPLE_TRANSACTIONS.keys())}")
        sys.exit(1)

    transaction = FraudTransaction(**tx_data)

    # Load detector
    print("[pipeline] Loading detector model...")
    detector = FraudDetector.load()
    print(f"[pipeline] Model loaded (version={detector.model_version})")

    # Run pipeline
    print(f"[pipeline] Running for {args.tx} (audience={args.audience})...")
    result = run_pipeline(
        transaction,
        detector=detector,
        target_audience=args.audience,
    )

    # Output
    print(f"\n{'='*60}")
    print(f"Pipeline Result — {args.tx}")
    print(f"{'='*60}")
    print(f"Completed: {result.completed}")

    if result.error:
        print(f"Error:     {result.error} (stage: {result.error_stage})")

    if result.detection_result:
        dr = result.detection_result
        print("\n--- Detection ---")
        print(f"Fraud probability: {dr.fraud_probability:.4f}")
        print(f"Predicted fraud:   {dr.is_fraud_predicted}")
        print(f"Confidence tier:   {dr.confidence_tier}")
        print(f"Latency:           {dr.inference_latency_ms:.1f}ms")
        print("Top SHAP features:")
        for f in dr.top_shap_features:
            print(f"  {f.feature_name:20s} SHAP={f.shap_value:+.4f}  val={f.feature_value}")

    if result.explanation_result:
        er = result.explanation_result
        print(f"\n--- Explanation ({er.target_audience}) ---")
        print(f"Generated: {er.explanation_generated}")
        print(f"Text:\n{er.explanation_text}")
        print(f"Cited:     {er.cited_features}")
        print(f"Cost:      ${er.token_cost_usd:.6f}")

    if result.eval_result:
        ev = result.eval_result
        print("\n--- Evaluation ---")
        print(f"Overall:     {ev.overall_score:.2f}")
        print(f"Passed:      {ev.passed}")
        print(f"Grounding:   {ev.grounding_score:.2f}")
        print(f"Clarity:     {ev.clarity_score:.2f}")
        print(f"Completeness:{ev.completeness_score:.2f}")
        print(f"Audience:    {ev.audience_appropriateness_score:.2f}")
        if ev.failure_reasons:
            print(f"Failures:    {ev.failure_reasons}")

    # JSON dump for machine consumption
    output = {
        "transaction_id": result.transaction.TransactionID,
        "completed": result.completed,
        "error": result.error,
        "error_stage": result.error_stage,
    }
    if result.detection_result:
        output["detection_result"] = result.detection_result.model_dump()
    if result.explanation_result:
        output["explanation_result"] = result.explanation_result.model_dump()
    if result.eval_result:
        output["eval_result"] = result.eval_result.model_dump()

    print("\n--- JSON ---")
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
