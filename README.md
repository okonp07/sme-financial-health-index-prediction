# SME Financial Health Index Prediction

End-to-end machine learning project for predicting SME Financial Health Index (`Low`, `Medium`, `High`) from structured survey and business data. The workflow is designed for reproducibility, modularity, and strong multiclass F1 performance.

## Project layout

- `data/raw/`: provided Zindi source files
- `notebooks/`: starter notebook copy
- `src/data/`: data loading utilities
- `src/features/`: feature engineering and schema helpers
- `src/models/`: training, cross-validation, evaluation, and artifact saving
- `src/inference/`: reusable prediction helpers
- `outputs/eda/`: generated plots and EDA summaries
- `outputs/metrics/`: model comparison tables and evaluation reports
- `outputs/submissions/`: Zindi-ready prediction files
- `artifacts/`: trained model artifacts

## Modeling approach

- Normalizes inconsistent survey categories such as `Don't know` variants and account-status labels
- Engineers business health features:
  - log monetary features
  - turnover/expense/income ratios
  - business maturity features
  - access-to-finance counts
  - missingness and signal counts
- Benchmarks multiple models:
  - Logistic Regression
  - Random Forest
  - Extra Trees
  - LightGBM
  - CatBoost
  - XGBoost
- Uses stratified cross-validation with F1-focused model selection
- Tests a soft-voting ensemble of the top three models

## Metric strategy

Because the competition metric is described as F1 on an imbalanced multiclass task, the pipeline tracks:

- weighted F1
- macro F1
- micro F1
- confusion matrix
- per-class precision, recall, and F1

Model selection defaults to `weighted_f1`, while also considering `macro_f1` to avoid overfitting to the dominant class.

## Run the project

Generate EDA outputs:

```bash
python eda.py
```

Train models, save artifacts, and export a submission:

```bash
python train.py
```

Run inference with a saved artifact:

```bash
python predict.py --artifact artifacts/trained_pipeline.joblib --input data/raw/Test.csv --output outputs/submissions/prediction_output.csv
```

## Key outputs

- `outputs/eda/eda_report.md`
- `outputs/metrics/model_comparison.csv`
- `outputs/metrics/classification_report.json`
- `outputs/metrics/confusion_matrix.json`
- `artifacts/trained_pipeline.joblib`
- `outputs/submissions/submission.csv`

## Notes

- The `High` class is rare, so class weights and robust cross-validation are central to the approach.
- Country effects are strong enough to preserve location information explicitly in the feature set.
- The code uses only open-source libraries and keeps all preprocessing inside Python.
- Raw competition data, generated artifacts, and run outputs are kept local and excluded from git in the repo version of this project.
