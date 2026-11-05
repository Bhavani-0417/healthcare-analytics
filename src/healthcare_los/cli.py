"""Command-line entry point for model training."""

from __future__ import annotations

import argparse
import json

from healthcare_los.modeling import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a patient length-of-stay classifier."
    )
    parser.add_argument("--data", default="data/raw/train.csv", help="Path to labeled training CSV")
    parser.add_argument("--output", default="artifacts", help="Directory for model and evaluation outputs")
    parser.add_argument("--target", default="Stay", help="Name of the target column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not 0.05 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.05 and 0.5")

    metrics = train_and_save(
        data_path=args.data,
        output_dir=args.output,
        target_column=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2))
    print(f"\nSaved model and evaluation files to: {args.output}")
