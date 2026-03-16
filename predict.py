from __future__ import annotations

import argparse

import pandas as pd

from src.data.loaders import load_test_data
from src.inference.predict import load_artifact, predict_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for SME FHI prediction.")
    parser.add_argument(
        "--artifact",
        default="artifacts/trained_pipeline.joblib",
        help="Path to the saved training artifact.",
    )
    parser.add_argument(
        "--input",
        default="data/raw/Test.csv",
        help="Path to the CSV file to score.",
    )
    parser.add_argument(
        "--output",
        default="outputs/submissions/prediction_output.csv",
        help="Path to the output CSV file.",
    )
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    input_df = load_test_data() if args.input == "data/raw/Test.csv" else pd.read_csv(args.input)
    predictions = predict_dataframe(input_df, artifact)
    predictions.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
