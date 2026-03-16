from __future__ import annotations

import json

import joblib

from src.config import METRICS_DIR, SUBMISSIONS_DIR
from src.data.loaders import load_test_data, load_train_data
from src.inference.predict import predict_dataframe
from src.models.training import train_pipeline


def main() -> None:
    train_df = load_train_data()
    training_outputs = train_pipeline(train_df)
    saved_payload = training_outputs["artifact_path"]

    test_df = load_test_data()
    artifact = joblib.load(saved_payload)
    predictions = predict_dataframe(test_df, artifact)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions[["ID", "Target"]].to_csv(
        SUBMISSIONS_DIR / "submission.csv", index=False
    )
    predictions.to_csv(SUBMISSIONS_DIR / "submission_with_probabilities.csv", index=False)

    summary = {
        "selected_model": training_outputs["selection"]["best_model_name"],
        "best_single_model": training_outputs["selection"]["best_single_model_name"],
        "artifact_path": str(training_outputs["artifact_path"]),
        "submission_path": str(SUBMISSIONS_DIR / "submission.csv"),
    }
    (METRICS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
