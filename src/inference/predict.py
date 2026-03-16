from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import ID_COLUMN
from src.features.engineering import engineer_features
from src.models.training import (
    apply_probability_adjustments,
    apply_feature_augmenter,
    build_geo_meta_features,
    predict_country_specialist_probabilities,
    prepare_for_native_boosters,
)


def load_artifact(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def _predict_proba_single(model_name: str, model: Any, features: pd.DataFrame) -> np.ndarray:
    if model_name in {"lightgbm", "lightgbm_tuned", "catboost"}:
        prepared = prepare_for_native_boosters(features)
        return model.predict_proba(prepared)
    return model.predict_proba(features)


def predict_dataframe(input_df: pd.DataFrame, artifact_payload: dict[str, Any]) -> pd.DataFrame:
    ids = input_df[ID_COLUMN].copy()
    base_features = engineer_features(input_df)
    base_features = base_features.reindex(columns=artifact_payload["feature_columns"])

    model_artifact = artifact_payload["model_artifact"]
    feature_augmenter = model_artifact.get("feature_augmenter")
    features = apply_feature_augmenter(base_features, feature_augmenter) if feature_augmenter else base_features
    if model_artifact["artifact_type"] == "weighted_ensemble":
        member_probabilities = np.stack(
            [
                _predict_proba_single(name, model_artifact["models"][name], features)
                for name in model_artifact["members"]
            ],
            axis=0,
        )
        probabilities = np.tensordot(
            np.array(model_artifact["weights"], dtype=float),
            member_probabilities,
            axes=(0, 0),
        )
        probabilities = apply_probability_adjustments(
            probabilities,
            np.array(model_artifact["class_adjustments"], dtype=float),
        )
    elif model_artifact["artifact_type"] == "geo_weighted_blend":
        global_member_probabilities = np.stack(
            [
                _predict_proba_single(name, model_artifact["global_models"][name], features)
                for name in model_artifact["global_members"]
            ],
            axis=0,
        )
        global_probabilities = np.tensordot(
            np.array(model_artifact["global_weights"], dtype=float),
            global_member_probabilities,
            axes=(0, 0),
        )
        global_probabilities = apply_probability_adjustments(
            global_probabilities,
            np.array(model_artifact["global_class_adjustments"], dtype=float),
        )
        specialist_probabilities = predict_country_specialist_probabilities(
            features,
            model_artifact["specialist_models"],
        )
        probabilities = (
            model_artifact["blend_source_weights"][0] * global_probabilities
            + model_artifact["blend_source_weights"][1] * specialist_probabilities
        )
        probabilities = apply_probability_adjustments(
            probabilities,
            np.array(model_artifact["blend_class_adjustments"], dtype=float),
        )
    elif model_artifact["artifact_type"] == "geo_hybrid_ensemble":
        global_member_probabilities = np.stack(
            [
                _predict_proba_single(name, model_artifact["global_models"][name], features)
                for name in model_artifact["global_members"]
            ],
            axis=0,
        )
        global_probabilities = np.tensordot(
            np.array(model_artifact["global_weights"], dtype=float),
            global_member_probabilities,
            axes=(0, 0),
        )
        global_probabilities = apply_probability_adjustments(
            global_probabilities,
            np.array(model_artifact["global_class_adjustments"], dtype=float),
        )
        specialist_probabilities = predict_country_specialist_probabilities(
            features,
            model_artifact["specialist_models"],
        )
        meta_features = build_geo_meta_features(
            global_probabilities,
            specialist_probabilities,
            features["country"],
            model_artifact["country_order"],
        )
        probabilities = model_artifact["meta_model"].predict_proba(meta_features)
    elif model_artifact["artifact_type"] == "binary_guided_ensemble":
        global_member_probabilities = np.stack(
            [
                _predict_proba_single(name, model_artifact["global_models"][name], features)
                for name in model_artifact["global_members"]
            ],
            axis=0,
        )
        probabilities = np.tensordot(
            np.array(model_artifact["global_weights"], dtype=float),
            global_member_probabilities,
            axes=(0, 0),
        )
        probabilities = apply_probability_adjustments(
            probabilities,
            np.array(model_artifact["global_class_adjustments"], dtype=float),
        )
        high_vs_rest_prob = model_artifact["high_vs_rest_model"].predict_proba(features)[:, 1]
        high_vs_medium_prob = model_artifact["high_vs_medium_model"].predict_proba(features)[:, 1]
        params = model_artifact["binary_guidance_params"]

        label_encoder = artifact_payload["label_encoder"]
        class_to_idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}
        high_idx = class_to_idx["High"]
        low_idx = class_to_idx["Low"]
        medium_idx = class_to_idx["Medium"]

        adjusted = probabilities.copy()
        non_low_mass = np.clip(1.0 - adjusted[:, low_idx], 0.0, 1.0)
        guided_high = (1.0 - params["lambda_high"]) * adjusted[:, high_idx] + params["lambda_high"] * high_vs_rest_prob
        hm_high = non_low_mass * high_vs_medium_prob
        blended_high = (1.0 - params["lambda_hm"]) * guided_high + params["lambda_hm"] * hm_high
        blended_high = np.clip(blended_high * params["high_scale"], 0.0, 0.995)

        adjusted[:, high_idx] = blended_high
        adjusted[:, low_idx] = np.clip(adjusted[:, low_idx], 0.001, 0.995)
        adjusted[:, medium_idx] = np.clip(1.0 - adjusted[:, low_idx] - adjusted[:, high_idx], 0.001, 0.995)
        probabilities = adjusted / adjusted.sum(axis=1, keepdims=True)
    elif model_artifact["artifact_type"] == "class_specialist_ensemble":
        label_encoder = artifact_payload["label_encoder"]
        class_indices = {label: idx for idx, label in enumerate(label_encoder.classes_)}
        probabilities = np.zeros((len(features), len(label_encoder.classes_)))
        cached_probs = {}
        for class_label, model_name in model_artifact["class_specialist_map"].items():
            if model_name not in cached_probs:
                cached_probs[model_name] = _predict_proba_single(
                    model_name,
                    model_artifact["models"][model_name],
                    features,
                )
            class_idx = class_indices[class_label]
            probabilities[:, class_idx] = (
                cached_probs[model_name][:, class_idx] * model_artifact["class_specialist_scales"][class_label]
            )
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    elif model_artifact["artifact_type"] == "stacked_ensemble":
        meta_features = np.hstack(
            [
                _predict_proba_single(name, model_artifact["models"][name], features)
                for name in model_artifact["members"]
            ]
        )
        probabilities = model_artifact["meta_model"].predict_proba(meta_features)
    else:
        probabilities = _predict_proba_single(
            model_artifact["model_name"], model_artifact["model"], features
        )

    label_encoder = artifact_payload["label_encoder"]
    prediction_idx = probabilities.argmax(axis=1)
    predictions = label_encoder.inverse_transform(prediction_idx)

    prediction_frame = pd.DataFrame({"ID": ids, "Target": predictions})
    probability_frame = pd.DataFrame(
        probabilities,
        columns=[f"prob_{label}" for label in label_encoder.classes_],
    )
    return pd.concat([prediction_frame, probability_frame], axis=1)
