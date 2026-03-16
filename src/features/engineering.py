from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import ID_COLUMN, TARGET_COLUMN


MONETARY_COLUMNS = ["personal_income", "business_expenses", "business_turnover"]
ACCOUNT_STATUS_COLUMNS = [
    "motor_vehicle_insurance",
    "has_mobile_money",
    "has_credit_card",
    "has_loan_account",
    "has_internet_banking",
    "has_debit_card",
    "medical_insurance",
    "funeral_insurance",
    "uses_friends_family_savings",
    "uses_informal_lender",
]
YES_NO_COLUMNS = [
    "attitude_stable_business_environment",
    "attitude_worried_shutdown",
    "compliance_income_tax",
    "perception_insurance_doesnt_cover_losses",
    "perception_cannot_afford_insurance",
    "current_problem_cash_flow",
    "has_cellphone",
    "attitude_satisfied_with_achievement",
    "keeps_financial_records",
    "perception_insurance_companies_dont_insure_businesses_like_yours",
    "perception_insurance_important",
    "has_insurance",
    "covid_essential_service",
    "attitude_more_successful_next_year",
    "problem_sourcing_money",
    "marketing_word_of_mouth",
    "future_risk_theft_stock",
    "motivation_make_more_money",
]

STATUS_SCORE_MAP = {
    "never had": 0.0,
    "used to have but don't have now": 1.0,
    "have now": 2.0,
}
YES_NO_SCORE_MAP = {
    "no": 0.0,
    "yes": 1.0,
    "yes, sometimes": 0.5,
    "yes, always": 1.0,
}


def normalize_text_value(value: object) -> object:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if not text:
        return np.nan

    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("\u200e", "")
        .replace("?", "'")
    )
    text = re.sub(r"\s+", " ", text)
    lowered = text.lower()

    if lowered == "0":
        return "No"
    if "don't know" in lowered or "dont know" in lowered or "do not know" in lowered:
        return "Don't know"
    if "n/a" in lowered:
        return "Don't know"
    if "doesn't apply" in lowered or "dont apply" in lowered:
        return "Don't know"
    if lowered == "refused":
        return "Refused"
    if lowered == "yes":
        return "Yes"
    if lowered == "no":
        return "No"
    if lowered == "have now":
        return "Have now"
    if lowered == "never had":
        return "Never had"
    if lowered == "used to have but don't have now":
        return "Used to have but don't have now"
    if lowered == "yes, always":
        return "Yes, always"
    if lowered == "yes, sometimes":
        return "Yes, sometimes"
    return text


def normalize_categorical_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = df.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].map(normalize_text_value)
    return frame


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.div(denominator.replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    object_columns = frame.select_dtypes(include="object").columns
    frame = normalize_categorical_columns(frame, object_columns)

    frame["business_age_total_months"] = (
        frame["business_age_years"].fillna(0) * 12 + frame["business_age_months"].fillna(0)
    )
    frame["business_age_total_years"] = frame["business_age_total_months"] / 12

    for column in MONETARY_COLUMNS + ["business_age_total_months"]:
        if column in frame.columns:
            frame[f"{column}_log1p"] = np.log1p(frame[column].clip(lower=0))

    frame["turnover_to_expense_ratio"] = _safe_divide(
        frame["business_turnover"], frame["business_expenses"]
    )
    frame["turnover_to_income_ratio"] = _safe_divide(
        frame["business_turnover"], frame["personal_income"]
    )
    frame["expense_to_income_ratio"] = _safe_divide(
        frame["business_expenses"], frame["personal_income"]
    )
    frame["profit_proxy"] = frame["business_turnover"] - frame["business_expenses"]
    frame["profit_margin_proxy"] = _safe_divide(frame["profit_proxy"], frame["business_turnover"])
    frame["turnover_per_month_active"] = _safe_divide(
        frame["business_turnover"], frame["business_age_total_months"]
    )
    frame["expenses_per_month_active"] = _safe_divide(
        frame["business_expenses"], frame["business_age_total_months"]
    )

    frame["owner_age_band"] = pd.cut(
        frame["owner_age"],
        bins=[0, 29, 39, 49, 59, 120],
        labels=["18-29", "30-39", "40-49", "50-59", "60+"],
        include_lowest=True,
    ).astype("object")
    frame["business_maturity_band"] = pd.cut(
        frame["business_age_total_years"],
        bins=[-1, 1, 3, 7, 15, 100],
        labels=["new", "early", "growing", "established", "legacy"],
        include_lowest=True,
    ).astype("object")

    status_scores = {}
    for column in ACCOUNT_STATUS_COLUMNS:
        if column in frame.columns:
            normalized = frame[column].str.lower()
            status_scores[column] = normalized.map(STATUS_SCORE_MAP)
            frame[f"{column}_status_score"] = status_scores[column]

    if status_scores:
        status_matrix = pd.DataFrame(status_scores, index=frame.index)
        frame["access_status_score_mean"] = status_matrix.mean(axis=1)
        frame["access_products_current_count"] = (status_matrix == 2.0).sum(axis=1)
        frame["access_products_past_count"] = (status_matrix == 1.0).sum(axis=1)
        frame["access_products_never_count"] = (status_matrix == 0.0).sum(axis=1)

    yes_no_scores = {}
    for column in YES_NO_COLUMNS:
        if column in frame.columns:
            normalized = frame[column].str.lower()
            yes_no_scores[column] = normalized.map(YES_NO_SCORE_MAP)
            frame[f"{column}_score"] = yes_no_scores[column]

    if yes_no_scores:
        yn_matrix = pd.DataFrame(yes_no_scores, index=frame.index)
        frame["positive_signal_count"] = (yn_matrix >= 1.0).sum(axis=1)
        frame["negative_signal_count"] = (yn_matrix == 0.0).sum(axis=1)
        frame["ambiguous_signal_count"] = yn_matrix.isna().sum(axis=1)

    frame["record_missing_count"] = frame.isna().sum(axis=1)
    frame["record_missing_ratio"] = frame.isna().mean(axis=1)

    columns_to_drop = [column for column in [ID_COLUMN, TARGET_COLUMN] if column in frame.columns]
    return frame.drop(columns=columns_to_drop)


def get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns
