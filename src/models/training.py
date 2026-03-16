from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import ARTIFACTS_DIR, METRICS_DIR, N_SPLITS, RANDOM_STATE
from src.features.engineering import engineer_features, get_feature_types


PRIMARY_METRIC = "weighted_f1"
COUNTRY_SPECIALIST_MODEL = "xgboost_tuned"
COUNTRY_ONLY_MIN_CLASS_COUNT = 12
COUNTRY_WEIGHT_MULTIPLIER = 4.0


@dataclass
class CVResult:
    name: str
    fold_scores: list[dict[str, float]]
    oof_predictions: np.ndarray
    oof_probabilities: np.ndarray
    summary: dict[str, float]


@dataclass
class GeographyResult:
    result: CVResult
    specialist_metadata: dict[str, Any]


@dataclass
class FeatureAugmenter:
    categorical_columns: list[str]
    encoding_columns: list[str]
    numeric_country_columns: list[str]
    country_frequency_map: dict[str, float]
    frequency_maps: dict[str, dict[str, float]]
    class_target_maps: dict[str, dict[str, dict[str, float]]]
    global_class_target_defaults: dict[str, float]
    country_numeric_means: dict[str, dict[str, float]]
    country_numeric_stds: dict[str, dict[str, float]]
    country_numeric_medians: dict[str, dict[str, float]]
    country_numeric_sorted_values: dict[str, dict[str, list[float]]]


@dataclass
class BinaryGuidanceResult:
    result: CVResult
    high_vs_rest_oof: np.ndarray
    high_vs_medium_oof: np.ndarray
    params: dict[str, Any]


def build_sklearn_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns, categorical_columns = get_feature_types(feature_frame)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        # Some country-specific folds have numeric columns that are entirely missing.
                        # Keep those columns in the pipeline instead of dropping them at fit time.
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=0.01),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def prepare_for_native_boosters(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    numeric_columns, categorical_columns = get_feature_types(prepared)
    for column in categorical_columns:
        prepared[column] = prepared[column].fillna("Missing").astype(str).astype("category")
    for column in numeric_columns:
        prepared[column] = prepared[column].replace([np.inf, -np.inf], np.nan)
        prepared[column] = prepared[column].astype(float)
    return prepared


def create_model_factories(feature_frame: pd.DataFrame) -> dict[str, Any]:
    preprocessor = build_sklearn_preprocessor(feature_frame)

    return {
        "logistic_regression": lambda: Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2500,
                        class_weight="balanced",
                        C=0.7,
                        multi_class="auto",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": lambda: Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": lambda: Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "lightgbm": lambda: lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
        ),
        "catboost": lambda: CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            auto_class_weights="Balanced",
            iterations=400,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=5.0,
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
        "xgboost": lambda: Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        n_estimators=350,
                        learning_rate=0.04,
                        max_depth=6,
                        min_child_weight=2,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost_tuned": lambda: Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        n_estimators=450,
                        learning_rate=0.035,
                        max_depth=6,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_alpha=0.05,
                        reg_lambda=1.0,
                        tree_method="hist",
                        random_state=RANDOM_STATE + 7,
                    ),
                ),
            ]
        ),
        "lightgbm_tuned": lambda: lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=55,
            min_child_samples=25,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.2,
            random_state=RANDOM_STATE + 11,
            class_weight="balanced",
            verbose=-1,
        ),
    }


def create_binary_auxiliary_factory(feature_frame: pd.DataFrame) -> Any:
    preprocessor = build_sklearn_preprocessor(feature_frame)
    return lambda: Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=350,
                    learning_rate=0.04,
                    max_depth=5,
                    min_child_weight=2,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_alpha=0.05,
                    reg_lambda=1.0,
                    tree_method="hist",
                    random_state=RANDOM_STATE + 909,
                ),
            ),
        ]
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], label_encoder: LabelEncoder) -> dict[str, float]:
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class_map = {
        f"{label_name.lower()}_f1": float(per_class_f1[idx])
        for idx, label_name in enumerate(label_encoder.classes_)
    }
    return {
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        **per_class_map,
    }


def score_sort_key(summary: dict[str, float]) -> tuple[float, float, float]:
    return (
        summary["weighted_f1"],
        summary["macro_f1"],
        summary.get("high_f1", 0.0),
    )


def generate_weight_candidates(num_models: int, step: float = 0.05) -> list[np.ndarray]:
    if num_models == 1:
        return [np.array([1.0])]

    candidates: list[np.ndarray] = []
    units = int(round(1 / step))

    def recurse(position: int, remaining: int, current: list[int]) -> None:
        if position == num_models - 1:
            values = current + [remaining]
            candidates.append(np.array(values, dtype=float) / units)
            return
        for amount in range(remaining + 1):
            recurse(position + 1, remaining - amount, current + [amount])

    recurse(0, units, [])
    return candidates


def apply_probability_adjustments(probabilities: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    adjusted = probabilities * class_weights.reshape(1, -1)
    row_sums = adjusted.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return adjusted / row_sums


def search_weighted_ensemble(
    candidate_names: list[str],
    results: dict[str, CVResult],
    y: np.ndarray,
    labels: list[int],
    label_encoder: LabelEncoder,
) -> CVResult:
    candidate_probabilities = [results[name].oof_probabilities for name in candidate_names]
    best_summary: dict[str, float] | None = None
    best_probabilities: np.ndarray | None = None
    best_name = "weighted_ensemble_optimized"
    best_weights: np.ndarray | None = None
    best_class_adjustments: np.ndarray | None = None

    class_adjustment_grid = [
        np.array([high_scale, 1.0, medium_scale], dtype=float)
        for high_scale in [0.95, 1.0, 1.05, 1.1, 1.15]
        for medium_scale in [0.95, 1.0, 1.05]
    ]

    for weights in generate_weight_candidates(len(candidate_names), step=0.05):
        if np.count_nonzero(weights) < 2:
            continue
        blended = np.tensordot(weights, np.stack(candidate_probabilities, axis=0), axes=(0, 0))
        for class_adjustments in class_adjustment_grid:
            adjusted = apply_probability_adjustments(blended, class_adjustments)
            predictions = adjusted.argmax(axis=1)
            summary = compute_metrics(y, predictions, labels, label_encoder)
            if best_summary is None or score_sort_key(summary) > score_sort_key(best_summary):
                best_summary = summary
                best_probabilities = adjusted
                best_weights = weights
                best_class_adjustments = class_adjustments

    assert best_summary is not None
    assert best_probabilities is not None
    assert best_weights is not None
    assert best_class_adjustments is not None
    best_summary["weight_vector"] = [float(value) for value in best_weights]
    best_summary["class_adjustments"] = [float(value) for value in best_class_adjustments]

    return CVResult(
        name=best_name,
        fold_scores=[],
        oof_predictions=best_probabilities.argmax(axis=1),
        oof_probabilities=best_probabilities,
        summary=best_summary,
    )


def evaluate_stacker(
    candidate_names: list[str],
    results: dict[str, CVResult],
    y: np.ndarray,
    labels: list[int],
    label_encoder: LabelEncoder,
) -> CVResult:
    meta_features = np.hstack([results[name].oof_probabilities for name in candidate_names])
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + 101)
    oof_probabilities = np.zeros((len(y), len(label_encoder.classes_)))
    fold_scores: list[dict[str, float]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(meta_features, y), start=1):
        meta_model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.5,
            multi_class="auto",
            random_state=RANDOM_STATE + 101,
        )
        meta_model.fit(meta_features[train_idx], y[train_idx])
        probabilities = meta_model.predict_proba(meta_features[valid_idx])
        oof_probabilities[valid_idx] = probabilities
        predictions = probabilities.argmax(axis=1)
        fold_summary = compute_metrics(y[valid_idx], predictions, labels, label_encoder)
        fold_summary["fold"] = float(fold_idx)
        fold_scores.append(fold_summary)

    final_predictions = oof_probabilities.argmax(axis=1)
    summary = compute_metrics(y, final_predictions, labels, label_encoder)
    summary["stacking_members"] = candidate_names
    return CVResult(
        name="stacked_logistic_top_models",
        fold_scores=fold_scores,
        oof_predictions=final_predictions,
        oof_probabilities=oof_probabilities,
        summary=summary,
    )


def fit_model_by_name(
    model_name: str,
    factory: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Any:
    model = factory()
    if model_name in {"lightgbm", "lightgbm_tuned"}:
        X_train_prepared = prepare_for_native_boosters(X_train)
        model.fit(X_train_prepared, y_train, sample_weight=sample_weight)
        return model
    if model_name == "catboost":
        X_train_prepared = prepare_for_native_boosters(X_train)
        categorical_features = X_train_prepared.select_dtypes(include="category").columns.tolist()
        model.fit(X_train_prepared, y_train, cat_features=categorical_features)
        return model
    if model_name.startswith("xgboost") and sample_weight is not None:
        model.fit(X_train, y_train, model__sample_weight=sample_weight)
        return model
    model.fit(X_train, y_train)
    return model


def predict_proba_by_name(model_name: str, model: Any, X_frame: pd.DataFrame) -> np.ndarray:
    if model_name in {"lightgbm", "lightgbm_tuned", "catboost"}:
        return model.predict_proba(prepare_for_native_boosters(X_frame))
    return model.predict_proba(X_frame)


def fit_feature_augmenter(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    label_encoder: LabelEncoder,
) -> FeatureAugmenter:
    categorical_columns = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    encoding_columns = [
        column
        for column in categorical_columns
        if column != "country" and 2 <= X_train[column].nunique(dropna=False) <= 12
    ]

    numeric_country_columns = [
        column
        for column in [
            "personal_income_log1p",
            "business_expenses_log1p",
            "business_turnover_log1p",
            "profit_proxy",
            "profit_margin_proxy",
            "turnover_to_expense_ratio",
            "turnover_to_income_ratio",
            "expense_to_income_ratio",
            "business_age_total_years",
            "positive_signal_count",
            "negative_signal_count",
            "access_status_score_mean",
            "access_products_current_count",
            "record_missing_ratio",
        ]
        if column in X_train.columns
    ]

    y_series = pd.Series(label_encoder.inverse_transform(y_train), index=X_train.index, name="target_label")
    global_class_target_defaults = {
        class_name: float((y_series == class_name).mean()) for class_name in label_encoder.classes_
    }

    country_frequency_map = (
        X_train["country"].fillna("Missing").astype(str).value_counts(normalize=True).to_dict()
        if "country" in X_train.columns
        else {}
    )

    frequency_maps: dict[str, dict[str, float]] = {}
    class_target_maps: dict[str, dict[str, dict[str, float]]] = {}

    smoothing = 20.0
    for column in encoding_columns:
        series = X_train[column].fillna("Missing").astype(str)
        frequency_maps[column] = series.value_counts(normalize=True).to_dict()
        count_map = series.value_counts().to_dict()
        class_target_maps[column] = {}
        for class_name in label_encoder.classes_:
            target_binary = (y_series == class_name).astype(float)
            positive_sum = target_binary.groupby(series).sum()
            smoothed = {
                key: float(
                    (positive_sum.get(key, 0.0) + smoothing * global_class_target_defaults[class_name])
                    / (count_map[key] + smoothing)
                )
                for key in count_map
            }
            class_target_maps[column][class_name] = smoothed

    country_numeric_means: dict[str, dict[str, float]] = {}
    country_numeric_stds: dict[str, dict[str, float]] = {}
    country_numeric_medians: dict[str, dict[str, float]] = {}
    country_numeric_sorted_values: dict[str, dict[str, list[float]]] = {}

    if "country" in X_train.columns and numeric_country_columns:
        country_series = X_train["country"].fillna("Missing").astype(str)
        for country, idx in country_series.groupby(country_series).groups.items():
            country_slice = X_train.loc[idx, numeric_country_columns]
            country_numeric_means[country] = country_slice.mean().to_dict()
            std_values = country_slice.std().replace(0, np.nan)
            country_numeric_stds[country] = std_values.to_dict()
            country_numeric_medians[country] = country_slice.median().to_dict()
            country_numeric_sorted_values[country] = {
                column: np.sort(country_slice[column].dropna().to_numpy(dtype=float)).tolist()
                for column in numeric_country_columns
            }

    return FeatureAugmenter(
        categorical_columns=categorical_columns,
        encoding_columns=encoding_columns,
        numeric_country_columns=numeric_country_columns,
        country_frequency_map=country_frequency_map,
        frequency_maps=frequency_maps,
        class_target_maps=class_target_maps,
        global_class_target_defaults=global_class_target_defaults,
        country_numeric_means=country_numeric_means,
        country_numeric_stds=country_numeric_stds,
        country_numeric_medians=country_numeric_medians,
        country_numeric_sorted_values=country_numeric_sorted_values,
    )


def apply_feature_augmenter(X_frame: pd.DataFrame, augmenter: FeatureAugmenter) -> pd.DataFrame:
    augmented = X_frame.copy()
    derived_columns: dict[str, Any] = {}

    if "country" in augmented.columns:
        country_series = augmented["country"].fillna("Missing").astype(str)
        derived_columns["country_frequency"] = country_series.map(augmenter.country_frequency_map).fillna(0.0)
    else:
        country_series = pd.Series(["Missing"] * len(augmented), index=augmented.index)

    for column in augmenter.encoding_columns:
        series = augmented[column].fillna("Missing").astype(str)
        derived_columns[f"{column}_freq"] = series.map(augmenter.frequency_maps.get(column, {})).fillna(0.0)
        for class_name, mapping in augmenter.class_target_maps.get(column, {}).items():
            default_value = augmenter.global_class_target_defaults[class_name]
            derived_columns[f"{column}_te_{class_name.lower()}"] = series.map(mapping).fillna(default_value)

    for column in augmenter.numeric_country_columns:
        z_values = []
        median_gap_values = []
        percentile_values = []
        raw_values = augmented[column]
        for idx, value in raw_values.items():
            country = country_series.loc[idx]
            means = augmenter.country_numeric_means.get(country, {})
            stds = augmenter.country_numeric_stds.get(country, {})
            medians = augmenter.country_numeric_medians.get(country, {})
            sorted_values = augmenter.country_numeric_sorted_values.get(country, {}).get(column, [])
            mean_value = means.get(column, np.nan)
            std_value = stds.get(column, np.nan)
            median_value = medians.get(column, np.nan)

            if pd.isna(value):
                z_values.append(np.nan)
                median_gap_values.append(np.nan)
                percentile_values.append(np.nan)
                continue

            z_values.append((value - mean_value) / std_value if pd.notna(std_value) and std_value else np.nan)
            median_gap_values.append(value - median_value if pd.notna(median_value) else np.nan)

            if sorted_values:
                sorted_arr = np.asarray(sorted_values, dtype=float)
                percentile = float(np.searchsorted(sorted_arr, value, side="right") / len(sorted_arr))
            else:
                percentile = np.nan
            percentile_values.append(percentile)

        derived_columns[f"{column}_country_z"] = pd.Series(z_values, index=augmented.index)
        derived_columns[f"{column}_country_median_gap"] = pd.Series(median_gap_values, index=augmented.index)
        derived_columns[f"{column}_country_pct"] = pd.Series(percentile_values, index=augmented.index)

    if derived_columns:
        augmented = pd.concat([augmented, pd.DataFrame(derived_columns, index=augmented.index)], axis=1)

    return augmented


def build_geo_meta_features(
    global_probabilities: np.ndarray,
    specialist_probabilities: np.ndarray,
    country_series: pd.Series,
    country_order: list[str],
) -> np.ndarray:
    country_dummies = pd.get_dummies(country_series.astype(str)).reindex(columns=country_order, fill_value=0)
    confidence_gap = np.max(global_probabilities, axis=1, keepdims=True) - np.max(
        np.partition(global_probabilities, -2, axis=1)[:, -2:],
        axis=1,
        keepdims=True,
    )
    specialist_gap = np.max(specialist_probabilities, axis=1, keepdims=True) - np.max(
        np.partition(specialist_probabilities, -2, axis=1)[:, -2:],
        axis=1,
        keepdims=True,
    )
    return np.hstack(
        [
            global_probabilities,
            specialist_probabilities,
            specialist_probabilities - global_probabilities,
            confidence_gap,
            specialist_gap,
            country_dummies.to_numpy(dtype=float),
        ]
    )


def search_two_source_blend(
    result_name: str,
    first_probabilities: np.ndarray,
    second_probabilities: np.ndarray,
    y: np.ndarray,
    labels: list[int],
    label_encoder: LabelEncoder,
) -> CVResult:
    best_summary: dict[str, float] | None = None
    best_probabilities: np.ndarray | None = None
    best_weights: list[float] | None = None
    best_class_adjustments: list[float] | None = None

    for first_weight in np.arange(0.0, 1.01, 0.05):
        second_weight = 1.0 - first_weight
        if first_weight == 0.0 or second_weight == 0.0:
            continue
        blended = first_weight * first_probabilities + second_weight * second_probabilities
        for high_scale in [0.95, 1.0, 1.05, 1.1, 1.15]:
            adjusted = apply_probability_adjustments(blended, np.array([high_scale, 1.0, 1.0]))
            predictions = adjusted.argmax(axis=1)
            summary = compute_metrics(y, predictions, labels, label_encoder)
            if best_summary is None or score_sort_key(summary) > score_sort_key(best_summary):
                best_summary = summary
                best_probabilities = adjusted
                best_weights = [float(first_weight), float(second_weight)]
                best_class_adjustments = [float(high_scale), 1.0, 1.0]

    assert best_summary is not None
    assert best_probabilities is not None
    assert best_weights is not None
    assert best_class_adjustments is not None
    best_summary["source_weights"] = best_weights
    best_summary["class_adjustments"] = best_class_adjustments
    return CVResult(
        name=result_name,
        fold_scores=[],
        oof_predictions=best_probabilities.argmax(axis=1),
        oof_probabilities=best_probabilities,
        summary=best_summary,
    )


def optimize_binary_guided_probabilities(
    base_probabilities: np.ndarray,
    high_vs_rest_prob: np.ndarray,
    high_vs_medium_prob: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
) -> BinaryGuidanceResult:
    class_to_idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    high_idx = class_to_idx["High"]
    low_idx = class_to_idx["Low"]
    medium_idx = class_to_idx["Medium"]
    labels = list(range(len(label_encoder.classes_)))

    best_summary: dict[str, float] | None = None
    best_probabilities: np.ndarray | None = None
    best_params: dict[str, Any] | None = None

    for lambda_high in [0.1, 0.2, 0.3, 0.4]:
        for lambda_hm in [0.15, 0.25, 0.35, 0.45]:
            for high_scale in [1.0, 1.05, 1.1, 1.15]:
                adjusted = base_probabilities.copy()
                non_low_mass = np.clip(1.0 - adjusted[:, low_idx], 0.0, 1.0)
                guided_high = (1.0 - lambda_high) * adjusted[:, high_idx] + lambda_high * high_vs_rest_prob
                hm_high = non_low_mass * high_vs_medium_prob
                blended_high = (1.0 - lambda_hm) * guided_high + lambda_hm * hm_high
                blended_high = np.clip(blended_high * high_scale, 0.0, 0.995)

                adjusted[:, high_idx] = blended_high
                adjusted[:, low_idx] = np.clip(adjusted[:, low_idx], 0.001, 0.995)
                remaining = np.clip(1.0 - adjusted[:, low_idx] - adjusted[:, high_idx], 0.001, 0.995)
                adjusted[:, medium_idx] = remaining
                adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)

                predictions = adjusted.argmax(axis=1)
                summary = compute_metrics(y, predictions, labels, label_encoder)
                if best_summary is None or score_sort_key(summary) > score_sort_key(best_summary):
                    best_summary = summary
                    best_probabilities = adjusted
                    best_params = {
                        "lambda_high": lambda_high,
                        "lambda_hm": lambda_hm,
                        "high_scale": high_scale,
                    }

    assert best_summary is not None
    assert best_probabilities is not None
    assert best_params is not None
    best_summary.update(best_params)
    return BinaryGuidanceResult(
        result=CVResult(
            name="binary_guided_ensemble",
            fold_scores=[],
            oof_predictions=best_probabilities.argmax(axis=1),
            oof_probabilities=best_probabilities,
            summary=best_summary,
        ),
        high_vs_rest_oof=high_vs_rest_prob,
        high_vs_medium_oof=high_vs_medium_prob,
        params=best_params,
    )


def evaluate_high_auxiliary_models(
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    base_probabilities: np.ndarray,
) -> BinaryGuidanceResult:
    feature_augmenter = fit_feature_augmenter(X, y, label_encoder)
    augmented_schema = apply_feature_augmenter(X, feature_augmenter)
    high_model_factory = create_binary_auxiliary_factory(augmented_schema)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + 505)

    high_idx = int(np.where(label_encoder.classes_ == "High")[0][0])
    medium_idx = int(np.where(label_encoder.classes_ == "Medium")[0][0])

    high_vs_rest_oof = np.zeros(len(X))
    high_vs_medium_oof = np.zeros(len(X))

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"Evaluating high auxiliary models fold {fold_idx}/{N_SPLITS}...", flush=True)
        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y[train_idx]

        augmenter = fit_feature_augmenter(X_train, y_train, label_encoder)
        X_train_aug = apply_feature_augmenter(X_train, augmenter).reindex(columns=augmented_schema.columns)
        X_valid_aug = apply_feature_augmenter(X_valid, augmenter).reindex(columns=augmented_schema.columns)

        y_high = (y_train == high_idx).astype(int)
        high_model = high_model_factory()
        high_model.fit(X_train_aug, y_high)
        high_vs_rest_oof[valid_idx] = high_model.predict_proba(X_valid_aug)[:, 1]

        hm_mask = np.isin(y_train, [high_idx, medium_idx])
        y_hm = (y_train[hm_mask] == high_idx).astype(int)
        hm_model = high_model_factory()
        hm_model.fit(X_train_aug.iloc[hm_mask].copy(), y_hm)
        high_vs_medium_oof[valid_idx] = hm_model.predict_proba(X_valid_aug)[:, 1]

    return optimize_binary_guided_probabilities(
        base_probabilities=base_probabilities,
        high_vs_rest_prob=high_vs_rest_oof,
        high_vs_medium_prob=high_vs_medium_oof,
        y=y,
        label_encoder=label_encoder,
    )


def evaluate_class_specialist_ensemble(
    candidate_names: list[str],
    results: dict[str, CVResult],
    y: np.ndarray,
    label_encoder: LabelEncoder,
) -> CVResult:
    class_labels = label_encoder.classes_.tolist()
    class_indices = {label: idx for idx, label in enumerate(class_labels)}
    labels = list(range(len(class_labels)))

    class_model_map = {}
    for class_label in class_labels:
        metric_name = f"{class_label.lower()}_f1"
        class_model_map[class_label] = max(
            candidate_names,
            key=lambda name: (
                results[name].summary.get(metric_name, 0.0),
                results[name].summary["weighted_f1"],
                results[name].summary["macro_f1"],
            ),
        )

    combined = np.zeros((len(y), len(class_labels)))
    for class_label, model_name in class_model_map.items():
        class_idx = class_indices[class_label]
        combined[:, class_idx] = results[model_name].oof_probabilities[:, class_idx]

    best_summary: dict[str, float] | None = None
    best_probabilities: np.ndarray | None = None
    best_scales: dict[str, float] | None = None

    for high_scale in [0.95, 1.0, 1.05, 1.1, 1.15]:
        for medium_scale in [0.95, 1.0, 1.05, 1.1]:
            for low_scale in [0.95, 1.0, 1.05]:
                scales = {"High": high_scale, "Medium": medium_scale, "Low": low_scale}
                scaled = combined.copy()
                for class_label, scale in scales.items():
                    scaled[:, class_indices[class_label]] *= scale
                scaled = scaled / scaled.sum(axis=1, keepdims=True)
                predictions = scaled.argmax(axis=1)
                summary = compute_metrics(y, predictions, labels, label_encoder)
                if best_summary is None or score_sort_key(summary) > score_sort_key(best_summary):
                    best_summary = summary
                    best_probabilities = scaled
                    best_scales = scales

    assert best_summary is not None
    assert best_probabilities is not None
    assert best_scales is not None
    best_summary["class_model_map"] = class_model_map
    best_summary["class_scales"] = best_scales

    return CVResult(
        name="class_specialist_ensemble",
        fold_scores=[],
        oof_predictions=best_probabilities.argmax(axis=1),
        oof_probabilities=best_probabilities,
        summary=best_summary,
    )


def evaluate_geography_specialists(
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    global_probabilities: np.ndarray,
) -> GeographyResult:
    model_factories = create_model_factories(X)
    specialist_factory = model_factories[COUNTRY_SPECIALIST_MODEL]
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + 303)
    labels = list(range(len(label_encoder.classes_)))
    countries = sorted(X["country"].astype(str).unique().tolist())
    specialist_oof = np.zeros((len(X), len(label_encoder.classes_)))
    country_strategy_counter: dict[str, dict[str, int]] = {
        country: {"country_only": 0, "weighted_specialist": 0} for country in countries
    }

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"Evaluating geography specialists fold {fold_idx}/{N_SPLITS}...", flush=True)
        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y[train_idx]

        for country in countries:
            valid_country_mask = X_valid["country"].astype(str) == country
            if not valid_country_mask.any():
                continue

            train_country_mask = X_train["country"].astype(str) == country
            country_counts = np.bincount(
                y_train[train_country_mask.to_numpy()],
                minlength=len(label_encoder.classes_),
            )
            use_country_only = (
                train_country_mask.sum() > 0
                and len(np.unique(y_train[train_country_mask.to_numpy()])) == len(label_encoder.classes_)
                and country_counts.min() >= COUNTRY_ONLY_MIN_CLASS_COUNT
            )

            if use_country_only:
                country_strategy_counter[country]["country_only"] += 1
                model = fit_model_by_name(
                    COUNTRY_SPECIALIST_MODEL,
                    specialist_factory,
                    X_train.loc[train_country_mask].copy(),
                    y_train[train_country_mask.to_numpy()],
                )
            else:
                country_strategy_counter[country]["weighted_specialist"] += 1
                sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
                sample_weight = sample_weight * np.where(train_country_mask.to_numpy(), COUNTRY_WEIGHT_MULTIPLIER, 1.0)
                model = fit_model_by_name(
                    COUNTRY_SPECIALIST_MODEL,
                    specialist_factory,
                    X_train,
                    y_train,
                    sample_weight=sample_weight,
                )

            specialist_oof[valid_idx[valid_country_mask.to_numpy()]] = predict_proba_by_name(
                COUNTRY_SPECIALIST_MODEL,
                model,
                X_valid.loc[valid_country_mask].copy(),
            )

    specialist_result = search_two_source_blend(
        result_name="geo_weighted_blend",
        first_probabilities=global_probabilities,
        second_probabilities=specialist_oof,
        y=y,
        labels=labels,
        label_encoder=label_encoder,
    )
    specialist_result.summary["specialist_model_name"] = COUNTRY_SPECIALIST_MODEL
    specialist_result.summary["country_strategy_counter"] = country_strategy_counter

    country_order = countries
    meta_features = build_geo_meta_features(global_probabilities, specialist_oof, X["country"], country_order)
    meta_splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + 404)
    meta_oof = np.zeros((len(X), len(label_encoder.classes_)))
    meta_fold_scores: list[dict[str, float]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(meta_splitter.split(meta_features, y), start=1):
        meta_model = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=0.4,
            multi_class="auto",
            random_state=RANDOM_STATE + 404,
        )
        meta_model.fit(meta_features[train_idx], y[train_idx])
        probabilities = meta_model.predict_proba(meta_features[valid_idx])
        meta_oof[valid_idx] = probabilities
        predictions = probabilities.argmax(axis=1)
        fold_summary = compute_metrics(y[valid_idx], predictions, labels, label_encoder)
        fold_summary["fold"] = float(fold_idx)
        meta_fold_scores.append(fold_summary)

    geo_meta_predictions = meta_oof.argmax(axis=1)
    geo_meta_summary = compute_metrics(y, geo_meta_predictions, labels, label_encoder)
    geo_meta_summary["specialist_model_name"] = COUNTRY_SPECIALIST_MODEL
    geo_meta_summary["country_strategy_counter"] = country_strategy_counter
    geo_meta_result = CVResult(
        name="geo_hybrid_stacker",
        fold_scores=meta_fold_scores,
        oof_predictions=geo_meta_predictions,
        oof_probabilities=meta_oof,
        summary=geo_meta_summary,
    )

    return GeographyResult(
        result=geo_meta_result if score_sort_key(geo_meta_result.summary) > score_sort_key(specialist_result.summary) else specialist_result,
        specialist_metadata={
            "specialist_oof_probabilities": specialist_oof,
            "country_order": country_order,
            "country_strategy_counter": country_strategy_counter,
            "geo_weighted_blend_summary": specialist_result.summary,
            "geo_hybrid_stacker_summary": geo_meta_summary,
        },
    )


def cross_validate_models(X: pd.DataFrame, y: np.ndarray, label_encoder: LabelEncoder) -> tuple[dict[str, CVResult], dict[str, Any]]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    schema_augmenter = fit_feature_augmenter(X, y, label_encoder)
    augmented_schema = apply_feature_augmenter(X, schema_augmenter)
    model_factories = create_model_factories(augmented_schema)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    labels = list(range(len(label_encoder.classes_)))
    results: dict[str, CVResult] = {}

    for model_name, factory in model_factories.items():
        print(f"Evaluating {model_name}...", flush=True)
        fold_scores: list[dict[str, float]] = []
        oof_predictions = np.zeros(len(X), dtype=int)
        oof_probabilities = np.zeros((len(X), len(label_encoder.classes_)))

        for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
            print(f"  Fold {fold_idx}/{N_SPLITS}", flush=True)
            X_train = X.iloc[train_idx].copy()
            X_valid = X.iloc[valid_idx].copy()
            y_train = y[train_idx]
            y_valid = y[valid_idx]

            augmenter = fit_feature_augmenter(X_train, y_train, label_encoder)
            X_train_aug = apply_feature_augmenter(X_train, augmenter).reindex(
                columns=augmented_schema.columns
            )
            X_valid_aug = apply_feature_augmenter(X_valid, augmenter).reindex(
                columns=augmented_schema.columns
            )

            model = factory()
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            if model_name in {"lightgbm", "lightgbm_tuned"}:
                X_train_prepared = prepare_for_native_boosters(X_train_aug)
                X_valid_prepared = prepare_for_native_boosters(X_valid_aug)
                model.fit(X_train_prepared, y_train, sample_weight=sample_weight)
                probabilities = model.predict_proba(X_valid_prepared)
            elif model_name == "catboost":
                X_train_prepared = prepare_for_native_boosters(X_train_aug)
                X_valid_prepared = prepare_for_native_boosters(X_valid_aug)
                categorical_features = X_train_prepared.select_dtypes(include="category").columns.tolist()
                model.fit(X_train_prepared, y_train, cat_features=categorical_features)
                probabilities = model.predict_proba(X_valid_prepared)
            elif model_name.startswith("xgboost"):
                model.fit(X_train_aug, y_train)
                probabilities = model.predict_proba(X_valid_aug)
            else:
                model.fit(X_train_aug, y_train)
                probabilities = model.predict_proba(X_valid_aug)

            predictions = probabilities.argmax(axis=1)
            oof_predictions[valid_idx] = predictions
            oof_probabilities[valid_idx] = probabilities

            scores = compute_metrics(y_valid, predictions, labels, label_encoder)
            scores["fold"] = float(fold_idx)
            fold_scores.append(scores)

        summary = {
            metric: float(np.mean([fold_score[metric] for fold_score in fold_scores]))
            for metric in ["weighted_f1", "macro_f1", "micro_f1", "high_f1", "low_f1", "medium_f1"]
        }
        results[model_name] = CVResult(
            name=model_name,
            fold_scores=fold_scores,
            oof_predictions=oof_predictions,
            oof_probabilities=oof_probabilities,
            summary=summary,
        )

    best_single_model_name = max(
        results,
        key=lambda name: score_sort_key(results[name].summary),
    )

    sorted_models = sorted(
        results,
        key=lambda name: score_sort_key(results[name].summary),
        reverse=True,
    )
    top_models = sorted_models[:4]
    ensemble_probabilities = np.mean(
        [results[name].oof_probabilities for name in top_models[:3]],
        axis=0,
    )
    ensemble_predictions = ensemble_probabilities.argmax(axis=1)
    ensemble_metrics = compute_metrics(y, ensemble_predictions, labels, label_encoder)
    ensemble_metrics["members"] = top_models[:3]
    ensemble_result = CVResult(
        name="soft_voting_top3",
        fold_scores=[],
        oof_predictions=ensemble_predictions,
        oof_probabilities=ensemble_probabilities,
        summary=ensemble_metrics,
    )
    results["soft_voting_top3"] = ensemble_result

    optimized_ensemble = search_weighted_ensemble(top_models, results, y, labels, label_encoder)
    optimized_ensemble.summary["members"] = top_models
    results[optimized_ensemble.name] = optimized_ensemble

    stacked_result = evaluate_stacker(top_models, results, y, labels, label_encoder)
    results[stacked_result.name] = stacked_result

    geography_result = evaluate_geography_specialists(
        X=X,
        y=y,
        label_encoder=label_encoder,
        global_probabilities=results["weighted_ensemble_optimized"].oof_probabilities,
    )
    results[geography_result.result.name] = geography_result.result
    selection_geo_summary = geography_result.specialist_metadata

    binary_guidance = evaluate_high_auxiliary_models(
        X=X,
        y=y,
        label_encoder=label_encoder,
        base_probabilities=results["weighted_ensemble_optimized"].oof_probabilities,
    )
    results[binary_guidance.result.name] = binary_guidance.result

    specialist_candidates = [
        name for name in results.keys()
        if name in model_factories or name in {"soft_voting_top3", "weighted_ensemble_optimized", "binary_guided_ensemble"}
    ]
    base_specialist_candidates = [name for name in specialist_candidates if name in model_factories]
    class_specialist = evaluate_class_specialist_ensemble(
        candidate_names=base_specialist_candidates,
        results=results,
        y=y,
        label_encoder=label_encoder,
    )
    results[class_specialist.name] = class_specialist

    best_model_name = max(
        results,
        key=lambda name: score_sort_key(results[name].summary),
    )

    selection = {
        "best_single_model_name": best_single_model_name,
        "best_model_name": best_model_name,
        "top_models_for_ensemble": top_models,
        "optimized_ensemble_weights": results["weighted_ensemble_optimized"].summary["weight_vector"],
        "optimized_ensemble_class_adjustments": results["weighted_ensemble_optimized"].summary["class_adjustments"],
        "stacking_members": top_models,
        "country_specialist_model_name": COUNTRY_SPECIALIST_MODEL,
        "country_order": selection_geo_summary["country_order"],
        "country_strategy_counter": selection_geo_summary["country_strategy_counter"],
        "geo_weighted_blend_summary": selection_geo_summary["geo_weighted_blend_summary"],
        "geo_hybrid_stacker_summary": selection_geo_summary["geo_hybrid_stacker_summary"],
        "binary_guidance_params": binary_guidance.params,
        "class_specialist_map": class_specialist.summary["class_model_map"],
        "class_specialist_scales": class_specialist.summary["class_scales"],
    }
    return results, selection


def save_evaluation_outputs(
    results: dict[str, CVResult],
    selection: dict[str, Any],
    label_encoder: LabelEncoder,
    y_true: np.ndarray,
    original_train_df: pd.DataFrame,
) -> None:
    summary_rows = []
    for name, result in results.items():
        row = {"model": name}
        row.update(result.summary)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["weighted_f1", "macro_f1", "high_f1"], ascending=False
    )
    summary_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    fold_rows = []
    for name, result in results.items():
        for fold_score in result.fold_scores:
            fold_rows.append({"model": name, **fold_score})
    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(METRICS_DIR / "fold_scores.csv", index=False)

    best_result = results[selection["best_model_name"]]
    report = classification_report(
        y_true,
        best_result.oof_predictions,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, best_result.oof_predictions)

    (METRICS_DIR / "classification_report.json").write_text(json.dumps(report, indent=2))
    (METRICS_DIR / "confusion_matrix.json").write_text(
        json.dumps({"labels": label_encoder.classes_.tolist(), "matrix": confusion.tolist()}, indent=2)
    )
    (METRICS_DIR / "model_selection.json").write_text(json.dumps(selection, indent=2))

    subgroup_rows = []
    subgroup_frame = original_train_df[["country", "offers_credit_to_customers"]].copy()
    subgroup_frame["business_age_bucket"] = pd.cut(
        original_train_df["business_age_years"].fillna(-1),
        bins=[-1, 1, 3, 7, 15, 100],
        labels=["0-1", "2-3", "4-7", "8-15", "16+"],
    ).astype(str)
    subgroup_frame["y_true"] = label_encoder.inverse_transform(y_true)
    subgroup_frame["y_pred"] = label_encoder.inverse_transform(best_result.oof_predictions)

    for column in ["country", "offers_credit_to_customers", "business_age_bucket"]:
        for group_name, group_df in subgroup_frame.groupby(column, dropna=False):
            subgroup_rows.append(
                {
                    "segment": column,
                    "group": str(group_name),
                    "rows": int(len(group_df)),
                    "weighted_f1": float(
                        f1_score(group_df["y_true"], group_df["y_pred"], average="weighted")
                    ),
                    "macro_f1": float(
                        f1_score(group_df["y_true"], group_df["y_pred"], average="macro")
                    ),
                }
            )
    pd.DataFrame(subgroup_rows).to_csv(METRICS_DIR / "subgroup_analysis.csv", index=False)


def fit_final_model(
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    selected_model_name: str,
    ensemble_members: list[str],
    optimized_ensemble_weights: list[float],
    optimized_ensemble_class_adjustments: list[float],
    stacking_members: list[str],
    country_order: list[str],
    country_specialist_model_name: str,
    selected_model_summary: dict[str, Any],
    class_specialist_map: dict[str, str],
    class_specialist_scales: dict[str, float],
) -> dict[str, Any]:
    full_augmenter = fit_feature_augmenter(X, y, label_encoder)
    X_augmented = apply_feature_augmenter(X, full_augmenter)
    model_factories = create_model_factories(X_augmented)

    def fit_one(model_name: str) -> Any:
        model = model_factories[model_name]()
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        if model_name in {"lightgbm", "lightgbm_tuned"}:
            X_prepared = prepare_for_native_boosters(X_augmented)
            model.fit(X_prepared, y, sample_weight=sample_weight)
            return model
        if model_name == "catboost":
            X_prepared = prepare_for_native_boosters(X_augmented)
            categorical_features = X_prepared.select_dtypes(include="category").columns.tolist()
            model.fit(X_prepared, y, cat_features=categorical_features)
            return model
        if model_name.startswith("xgboost"):
            model.fit(X_augmented, y)
            return model
        model.fit(X_augmented, y)
        return model

    if selected_model_name in {"soft_voting_top3", "weighted_ensemble_optimized"}:
        models = {name: fit_one(name) for name in ensemble_members}
        artifact = {
            "artifact_type": "weighted_ensemble",
            "members": ensemble_members,
            "models": models,
            "weights": optimized_ensemble_weights if selected_model_name == "weighted_ensemble_optimized" else [1 / len(ensemble_members)] * len(ensemble_members),
            "class_adjustments": optimized_ensemble_class_adjustments if selected_model_name == "weighted_ensemble_optimized" else [1.0, 1.0, 1.0],
            "feature_augmenter": full_augmenter,
        }
    elif selected_model_name in {"geo_weighted_blend", "geo_hybrid_stacker"}:
        global_models = {name: fit_one(name) for name in ensemble_members}
        specialist_factory = model_factories[country_specialist_model_name]
        specialist_models: dict[str, dict[str, Any]] = {}
        class_count_threshold = COUNTRY_ONLY_MIN_CLASS_COUNT
        for country in country_order:
            country_mask = X["country"].astype(str) == country
            country_counts = np.bincount(
                y[country_mask.to_numpy()],
                minlength=len(np.unique(y)),
            )
            use_country_only = (
                country_mask.sum() > 0
                and len(np.unique(y[country_mask.to_numpy()])) == len(np.unique(y))
                and country_counts.min() >= class_count_threshold
            )
            if use_country_only:
                specialist_model = fit_model_by_name(
                    country_specialist_model_name,
                    specialist_factory,
                    X_augmented.loc[country_mask].copy(),
                    y[country_mask.to_numpy()],
                )
                specialist_models[country] = {
                    "mode": "country_only",
                    "model_name": country_specialist_model_name,
                    "model": specialist_model,
                }
            else:
                sample_weight = compute_sample_weight(class_weight="balanced", y=y)
                sample_weight = sample_weight * np.where(country_mask.to_numpy(), COUNTRY_WEIGHT_MULTIPLIER, 1.0)
                specialist_model = fit_model_by_name(
                    country_specialist_model_name,
                    specialist_factory,
                    X_augmented,
                    y,
                    sample_weight=sample_weight,
                )
                specialist_models[country] = {
                    "mode": "weighted_specialist",
                    "model_name": country_specialist_model_name,
                    "model": specialist_model,
                }

        global_probabilities = np.tensordot(
            np.array(optimized_ensemble_weights, dtype=float),
            np.stack(
                [_predict_probabilities_for_fitted_model(name, global_models[name], X_augmented) for name in ensemble_members],
                axis=0,
            ),
            axes=(0, 0),
        )
        global_probabilities = apply_probability_adjustments(
            global_probabilities,
            np.array(optimized_ensemble_class_adjustments, dtype=float),
        )
        specialist_probabilities = predict_country_specialist_probabilities(
            X_augmented,
            specialist_models,
        )

        artifact = {
            "artifact_type": "geo_weighted_blend" if selected_model_name == "geo_weighted_blend" else "geo_hybrid_ensemble",
            "global_members": ensemble_members,
            "global_models": global_models,
            "global_weights": optimized_ensemble_weights,
            "global_class_adjustments": optimized_ensemble_class_adjustments,
            "country_order": country_order,
            "specialist_model_name": country_specialist_model_name,
            "specialist_models": specialist_models,
            "feature_augmenter": full_augmenter,
        }
        if selected_model_name == "geo_weighted_blend":
            source_weights = selected_model_summary["source_weights"]
            class_adjustments = selected_model_summary["class_adjustments"]
            artifact["blend_source_weights"] = source_weights
            artifact["blend_class_adjustments"] = class_adjustments
        else:
            meta_features = build_geo_meta_features(
                global_probabilities,
                specialist_probabilities,
                X_augmented["country"],
                country_order,
            )
            meta_model = LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                C=0.4,
                multi_class="auto",
                random_state=RANDOM_STATE + 404,
            )
            meta_model.fit(meta_features, y)
            artifact["meta_model"] = meta_model
    elif selected_model_name == "stacked_logistic_top_models":
        models = {name: fit_one(name) for name in stacking_members}
        train_meta_features = np.hstack(
            [
                _predict_probabilities_for_fitted_model(name, models[name], X_augmented)
                for name in stacking_members
            ]
        )
        meta_model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.5,
            multi_class="auto",
            random_state=RANDOM_STATE + 101,
        )
        meta_model.fit(train_meta_features, y)
        artifact = {
            "artifact_type": "stacked_ensemble",
            "members": stacking_members,
            "models": models,
            "meta_model": meta_model,
            "feature_augmenter": full_augmenter,
        }
    elif selected_model_name == "binary_guided_ensemble":
        global_models = {name: fit_one(name) for name in ensemble_members}
        high_idx = int(np.where(label_encoder.classes_ == "High")[0][0])
        medium_idx = int(np.where(label_encoder.classes_ == "Medium")[0][0])

        binary_factory = create_binary_auxiliary_factory(X_augmented)
        high_model = binary_factory()
        high_model.fit(X_augmented, (y == high_idx).astype(int))

        hm_mask = np.isin(y, [high_idx, medium_idx])
        high_medium_model = binary_factory()
        high_medium_model.fit(X_augmented.iloc[hm_mask].copy(), (y[hm_mask] == high_idx).astype(int))

        artifact = {
            "artifact_type": "binary_guided_ensemble",
            "global_members": ensemble_members,
            "global_models": global_models,
            "global_weights": optimized_ensemble_weights,
            "global_class_adjustments": optimized_ensemble_class_adjustments,
            "high_vs_rest_model": high_model,
            "high_vs_medium_model": high_medium_model,
            "binary_guidance_params": selected_model_summary,
            "feature_augmenter": full_augmenter,
        }
    elif selected_model_name == "class_specialist_ensemble":
        unique_models = sorted(set(class_specialist_map.values()))
        specialist_models = {name: fit_one(name) for name in unique_models}
        artifact = {
            "artifact_type": "class_specialist_ensemble",
            "models": specialist_models,
            "class_specialist_map": class_specialist_map,
            "class_specialist_scales": class_specialist_scales,
            "feature_augmenter": full_augmenter,
        }
    else:
        artifact = {
            "artifact_type": "single_model",
            "model_name": selected_model_name,
            "model": fit_one(selected_model_name),
            "feature_augmenter": full_augmenter,
        }

    return artifact


def _predict_probabilities_for_fitted_model(model_name: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    return predict_proba_by_name(model_name, model, X)


def predict_country_specialist_probabilities(
    X: pd.DataFrame,
    specialist_models: dict[str, dict[str, Any]],
) -> np.ndarray:
    sample_country = next(iter(specialist_models.values()))
    first_probs = predict_proba_by_name(
        sample_country["model_name"],
        sample_country["model"],
        X.iloc[[0]].copy(),
    )
    probabilities = np.zeros((len(X), first_probs.shape[1]))
    for country, spec in specialist_models.items():
        country_mask = X["country"].astype(str) == country
        if not country_mask.any():
            continue
        probabilities[country_mask.to_numpy()] = predict_proba_by_name(
            spec["model_name"],
            spec["model"],
            X.loc[country_mask].copy(),
        )
    return probabilities


def save_training_artifacts(
    model_artifact: dict[str, Any],
    label_encoder: LabelEncoder,
    feature_columns: list[str],
    output_path: Path | None = None,
) -> Path:
    artifact_path = output_path or ARTIFACTS_DIR / "trained_pipeline.joblib"
    payload = {
        "model_artifact": model_artifact,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
    }
    joblib.dump(payload, artifact_path)
    return artifact_path


def train_pipeline(train_df: pd.DataFrame) -> dict[str, Any]:
    feature_frame = engineer_features(train_df)
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(train_df["Target"])

    results, selection = cross_validate_models(feature_frame, target, label_encoder)
    save_evaluation_outputs(results, selection, label_encoder, target, train_df)

    final_artifact = fit_final_model(
        X=feature_frame,
        y=target,
        label_encoder=label_encoder,
        selected_model_name=selection["best_model_name"],
        ensemble_members=selection["top_models_for_ensemble"],
        optimized_ensemble_weights=selection["optimized_ensemble_weights"],
        optimized_ensemble_class_adjustments=selection["optimized_ensemble_class_adjustments"],
        stacking_members=selection["stacking_members"],
        country_order=selection["country_order"],
        country_specialist_model_name=selection["country_specialist_model_name"],
        selected_model_summary=results[selection["best_model_name"]].summary,
        class_specialist_map=selection["class_specialist_map"],
        class_specialist_scales=selection["class_specialist_scales"],
    )
    artifact_path = save_training_artifacts(
        model_artifact=final_artifact,
        label_encoder=label_encoder,
        feature_columns=feature_frame.columns.tolist(),
    )

    return {
        "artifact_path": artifact_path,
        "results": results,
        "selection": selection,
        "label_encoder": label_encoder,
        "feature_frame": feature_frame,
    }
