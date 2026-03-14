"""Run full pipeline for a single transaction.
Usage: poetry run python scripts/run_pipeline.py --tx TX_TEST_001
Implemented in Phase 5.
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", required=True)
    args = parser.parse_args()
    print(f"[pipeline] Running for {args.tx} — implement in Phase 5")

if __name__ == "__main__":
    main()
