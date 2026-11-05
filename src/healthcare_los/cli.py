"""Command-line model training entry point."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from healthcare_los.modeling import train_evaluate

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a patient LOS classifier.")
    parser.add_argument("--input", required=True, help="Path to a labeled CSV.")
    parser.add_argument("--target", default="Stay")
    parser.add_argument("--output", default="models/healthcare_los_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_file():
        parser.error(f"Input file does not exist: {source}")
    result = train_evaluate(pd.read_csv(source), args.target, args.test_size, args.random_state)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["bundle"], destination)
    print("Hold-out evaluation")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}")
    print(f"Model saved to: {destination}")

if __name__ == "__main__":
    main()
