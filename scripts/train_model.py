"""Train XGBoost model on IEEE-CIS data.
Usage: poetry run python scripts/train_model.py --sample 10000
Implemented in Phase 2.
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=10000)
    args = parser.parse_args()
    print(f"[train] Training on {args.sample} samples — implement in Phase 2")

if __name__ == "__main__":
    main()
