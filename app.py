from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.config import ID_COLUMN
from src.features.engineering import normalize_text_value
from src.inference.predict import load_artifact, predict_dataframe


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_PATH = APP_ROOT / "artifacts" / "trained_pipeline.joblib"
TRAIN_PATH = APP_ROOT / "data" / "raw" / "Train.csv"
TEST_PATH = APP_ROOT / "data" / "raw" / "Test.csv"
VARIABLE_DEFINITIONS_PATH = APP_ROOT / "data" / "raw" / "VariableDefinitions.csv"
RUN_SUMMARY_PATH = APP_ROOT / "outputs" / "metrics" / "run_summary.json"
EDA_SUMMARY_PATH = APP_ROOT / "outputs" / "eda" / "eda_summary.json"
EDA_REPORT_PATH = APP_ROOT / "outputs" / "eda" / "eda_report.md"
MODEL_COMPARISON_PATH = APP_ROOT / "outputs" / "metrics" / "model_comparison.csv"
SUBGROUP_ANALYSIS_PATH = APP_ROOT / "outputs" / "metrics" / "subgroup_analysis.csv"
CLASSIFICATION_REPORT_PATH = APP_ROOT / "outputs" / "metrics" / "classification_report.json"
EDA_IMAGE_DIR = APP_ROOT / "outputs" / "eda"

RAW_INPUT_COLUMNS = [
    "ID",
    "country",
    "owner_age",
    "attitude_stable_business_environment",
    "attitude_worried_shutdown",
    "compliance_income_tax",
    "perception_insurance_doesnt_cover_losses",
    "perception_cannot_afford_insurance",
    "personal_income",
    "business_expenses",
    "business_turnover",
    "business_age_years",
    "motor_vehicle_insurance",
    "has_mobile_money",
    "current_problem_cash_flow",
    "has_cellphone",
    "owner_sex",
    "offers_credit_to_customers",
    "attitude_satisfied_with_achievement",
    "has_credit_card",
    "keeps_financial_records",
    "perception_insurance_companies_dont_insure_businesses_like_yours",
    "perception_insurance_important",
    "has_insurance",
    "covid_essential_service",
    "attitude_more_successful_next_year",
    "problem_sourcing_money",
    "marketing_word_of_mouth",
    "has_loan_account",
    "has_internet_banking",
    "has_debit_card",
    "future_risk_theft_stock",
    "business_age_months",
    "medical_insurance",
    "funeral_insurance",
    "motivation_make_more_money",
    "uses_friends_family_savings",
    "uses_informal_lender",
]

NUMERIC_FIELDS = {
    "owner_age",
    "personal_income",
    "business_expenses",
    "business_turnover",
    "business_age_years",
    "business_age_months",
}

NUMERIC_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "owner_age": (18, 120),
    "personal_income": (0, None),
    "business_expenses": (0, None),
    "business_turnover": (0, None),
    "business_age_years": (0, None),
    "business_age_months": (0, 11),
}

DEFAULT_NUMERIC_VALUES: dict[str, float] = {
    "owner_age": 40.0,
    "personal_income": 2000.0,
    "business_expenses": 3000.0,
    "business_turnover": 6000.0,
    "business_age_years": 4.0,
    "business_age_months": 3.0,
}

OPTION_SETS = {
    "country": ["eswatini", "lesotho", "malawi", "zimbabwe"],
    "yes_no": ["No", "Yes"],
    "yes_no_dk": ["No", "Yes", "Don't know"],
    "yes_no_dk_refused": ["No", "Yes", "Don't know", "Refused"],
    "status": ["Never had", "Used to have but don't have now", "Have now", "Don't know"],
    "credit": ["No", "Yes, sometimes", "Yes, always"],
    "records": ["No", "Yes", "Yes, sometimes", "Yes, always"],
    "owner_sex": ["Female", "Male"],
}

FIELD_OPTION_GROUPS = {
    "country": "country",
    "attitude_stable_business_environment": "yes_no_dk",
    "attitude_worried_shutdown": "yes_no_dk",
    "compliance_income_tax": "yes_no_dk_refused",
    "perception_insurance_doesnt_cover_losses": "yes_no_dk",
    "perception_cannot_afford_insurance": "yes_no_dk",
    "motor_vehicle_insurance": "status",
    "has_mobile_money": "status",
    "current_problem_cash_flow": "yes_no",
    "has_cellphone": "yes_no",
    "owner_sex": "owner_sex",
    "offers_credit_to_customers": "credit",
    "attitude_satisfied_with_achievement": "yes_no_dk",
    "has_credit_card": "status",
    "keeps_financial_records": "records",
    "perception_insurance_companies_dont_insure_businesses_like_yours": "yes_no_dk",
    "perception_insurance_important": "yes_no_dk",
    "has_insurance": "yes_no",
    "covid_essential_service": "yes_no_dk",
    "attitude_more_successful_next_year": "yes_no_dk",
    "problem_sourcing_money": "yes_no",
    "marketing_word_of_mouth": "yes_no",
    "has_loan_account": "status",
    "has_internet_banking": "status",
    "has_debit_card": "status",
    "future_risk_theft_stock": "yes_no",
    "medical_insurance": "status",
    "funeral_insurance": "status",
    "motivation_make_more_money": "yes_no",
    "uses_friends_family_savings": "status",
    "uses_informal_lender": "status",
}

FIELD_GROUPS = {
    "Business profile": [
        "ID",
        "country",
        "owner_age",
        "owner_sex",
        "business_age_years",
        "business_age_months",
    ],
    "Financial picture": [
        "personal_income",
        "business_expenses",
        "business_turnover",
        "keeps_financial_records",
        "offers_credit_to_customers",
        "current_problem_cash_flow",
    ],
    "Owner outlook and operating context": [
        "attitude_stable_business_environment",
        "attitude_worried_shutdown",
        "attitude_satisfied_with_achievement",
        "attitude_more_successful_next_year",
        "problem_sourcing_money",
        "marketing_word_of_mouth",
        "covid_essential_service",
        "future_risk_theft_stock",
        "motivation_make_more_money",
    ],
    "Insurance and formal finance access": [
        "compliance_income_tax",
        "has_cellphone",
        "has_insurance",
        "motor_vehicle_insurance",
        "medical_insurance",
        "funeral_insurance",
        "has_mobile_money",
        "has_credit_card",
        "has_loan_account",
        "has_internet_banking",
        "has_debit_card",
    ],
    "Risk perceptions and informal finance": [
        "perception_insurance_doesnt_cover_losses",
        "perception_cannot_afford_insurance",
        "perception_insurance_companies_dont_insure_businesses_like_yours",
        "perception_insurance_important",
        "uses_friends_family_savings",
        "uses_informal_lender",
    ],
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(224, 107, 68, 0.14), transparent 24%),
                radial-gradient(circle at bottom left, rgba(44, 95, 122, 0.12), transparent 26%),
                linear-gradient(180deg, #f5efe5 0%, #fcfaf6 100%);
        }
        .hero-card {
            background: linear-gradient(135deg, #16324a 0%, #245f7a 56%, #d56c47 140%);
            border-radius: 26px;
            padding: 2rem 2rem 1.6rem 2rem;
            color: #fff7ee;
            margin-bottom: 1rem;
            box-shadow: 0 18px 42px rgba(22, 50, 74, 0.2);
        }
        .hero-card h1 {
            margin: 0 0 0.4rem 0;
            font-size: 2.4rem;
            line-height: 1.05;
            letter-spacing: -0.02em;
        }
        .hero-card p {
            margin: 0;
            max-width: 52rem;
            line-height: 1.55;
            color: rgba(255, 247, 238, 0.94);
        }
        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1rem;
        }
        .hero-pill {
            border: 1px solid rgba(255, 247, 238, 0.22);
            background: rgba(255, 247, 238, 0.1);
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.85rem;
        }
        .section-note {
            border-left: 4px solid #245f7a;
            background: rgba(36, 95, 122, 0.08);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin: 0.75rem 0 1rem 0;
        }
        .story-card {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(22, 50, 74, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 26px rgba(22, 50, 74, 0.06);
        }
        .story-card h4 {
            margin: 0 0 0.45rem 0;
            color: #16324a;
        }
        .story-card p {
            margin: 0;
            color: #31424d;
            line-height: 1.55;
        }
        .story-band {
            background: linear-gradient(135deg, rgba(213, 108, 71, 0.11), rgba(36, 95, 122, 0.08));
            border-radius: 22px;
            padding: 1.2rem 1.25rem;
            border: 1px solid rgba(213, 108, 71, 0.14);
            margin-bottom: 1rem;
        }
        .story-band h3 {
            margin: 0 0 0.4rem 0;
            color: #16324a;
        }
        .story-band p {
            margin: 0;
            color: #344754;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pretty_name(field_name: str) -> str:
    return field_name.replace("_", " ").strip().title()


def format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    numeric_value = float(value)
    return str(int(numeric_value)) if numeric_value.is_integer() else f"{numeric_value:.2f}"


@st.cache_resource(show_spinner=False)
def load_artifact_from_path(path_str: str) -> dict[str, Any]:
    return load_artifact(path_str)


@st.cache_resource(show_spinner=False)
def load_artifact_from_bytes(blob: bytes) -> dict[str, Any]:
    return joblib.load(BytesIO(blob))


@st.cache_data(show_spinner=False)
def load_variable_definitions() -> dict[str, str]:
    if not VARIABLE_DEFINITIONS_PATH.exists():
        return {}
    definitions_df = pd.read_csv(VARIABLE_DEFINITIONS_PATH)
    return dict(
        zip(definitions_df["VARIABLE_NAME"].astype(str), definitions_df["VARIABLE_DESCRIPTION"].astype(str))
    )


@st.cache_data(show_spinner=False)
def load_reference_inputs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if TRAIN_PATH.exists():
        frames.append(pd.read_csv(TRAIN_PATH).drop(columns=["Target"], errors="ignore"))
    if TEST_PATH.exists():
        frames.append(pd.read_csv(TEST_PATH))
    if not frames:
        return pd.DataFrame(columns=RAW_INPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False)


@st.cache_data(show_spinner=False)
def load_run_summary() -> dict[str, Any]:
    if not RUN_SUMMARY_PATH.exists():
        return {}
    return json.loads(RUN_SUMMARY_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_eda_summary() -> dict[str, Any]:
    if not EDA_SUMMARY_PATH.exists():
        return {}
    return json.loads(EDA_SUMMARY_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_classification_report() -> dict[str, Any]:
    if not CLASSIFICATION_REPORT_PATH.exists():
        return {}
    return json.loads(CLASSIFICATION_REPORT_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    if not MODEL_COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_subgroup_analysis() -> pd.DataFrame:
    if not SUBGROUP_ANALYSIS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SUBGROUP_ANALYSIS_PATH)


@st.cache_data(show_spinner=False)
def load_eda_report_text() -> str:
    if not EDA_REPORT_PATH.exists():
        return ""
    return EDA_REPORT_PATH.read_text()


def resolve_categorical_options(field_name: str, reference_df: pd.DataFrame) -> list[str]:
    option_group = FIELD_OPTION_GROUPS.get(field_name)
    if option_group:
        return OPTION_SETS[option_group]

    if field_name not in reference_df.columns:
        return []

    normalized = (
        reference_df[field_name]
        .map(normalize_text_value)
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    return normalized


def resolve_default_value(field_name: str, reference_df: pd.DataFrame) -> str | float | None:
    if field_name == ID_COLUMN:
        return "APP_RECORD_001"

    if field_name in NUMERIC_FIELDS:
        if field_name in reference_df.columns:
            values = pd.to_numeric(reference_df[field_name], errors="coerce").dropna()
            if not values.empty:
                return float(values.median())
        return DEFAULT_NUMERIC_VALUES[field_name]

    options = resolve_categorical_options(field_name, reference_df)
    if field_name in reference_df.columns:
        normalized = reference_df[field_name].map(normalize_text_value).dropna().astype(str)
        if not normalized.empty:
            mode = normalized.mode()
            if not mode.empty and mode.iloc[0] in options:
                return mode.iloc[0]
    return options[0] if options else None


@st.cache_data(show_spinner=False)
def build_field_profiles() -> dict[str, dict[str, Any]]:
    reference_df = load_reference_inputs()
    descriptions = load_variable_definitions()
    profiles: dict[str, dict[str, Any]] = {}

    for field_name in RAW_INPUT_COLUMNS:
        profiles[field_name] = {
            "label": pretty_name(field_name),
            "description": descriptions.get(field_name, pretty_name(field_name)),
            "kind": "numeric" if field_name in NUMERIC_FIELDS else ("text" if field_name == ID_COLUMN else "categorical"),
            "options": resolve_categorical_options(field_name, reference_df),
            "default": resolve_default_value(field_name, reference_df),
        }
    return profiles


@st.cache_data(show_spinner=False)
def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def build_blank_template() -> pd.DataFrame:
    template_row = {column: "" for column in RAW_INPUT_COLUMNS}
    template_row[ID_COLUMN] = "APP_TEMPLATE_001"
    return pd.DataFrame([template_row])


def build_example_template(field_profiles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    example_row = {}
    for field_name in RAW_INPUT_COLUMNS:
        profile = field_profiles[field_name]
        if profile["kind"] == "numeric":
            example_row[field_name] = profile["default"]
        elif field_name == ID_COLUMN:
            example_row[field_name] = "APP_EXAMPLE_001"
        else:
            example_row[field_name] = profile["default"] or ""
    return pd.DataFrame([example_row])


def parse_optional_numeric(field_name: str, raw_value: str) -> tuple[float | None, str | None]:
    cleaned = raw_value.strip()
    if not cleaned:
        return None, None

    try:
        parsed_value = float(cleaned.replace(",", ""))
    except ValueError:
        return None, f"`{pretty_name(field_name)}` must be a valid number."

    minimum, maximum = NUMERIC_BOUNDS[field_name]
    if minimum is not None and parsed_value < minimum:
        return None, f"`{pretty_name(field_name)}` cannot be below {minimum}."
    if maximum is not None and parsed_value > maximum:
        return None, f"`{pretty_name(field_name)}` cannot be above {maximum}."
    return parsed_value, None


def render_single_record_form(field_profiles: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    raw_values: dict[str, Any] = {}

    with st.form("single_record_form", clear_on_submit=False):
        st.markdown(
            '<div class="section-note">Use the training medians and most-common responses as a starting point, then adjust only the fields you know.</div>',
            unsafe_allow_html=True,
        )
        for group_name, group_fields in FIELD_GROUPS.items():
            with st.expander(group_name, expanded=group_name == "Business profile"):
                columns = st.columns(2)
                for index, field_name in enumerate(group_fields):
                    profile = field_profiles[field_name]
                    with columns[index % 2]:
                        if profile["kind"] == "text":
                            raw_values[field_name] = st.text_input(
                                profile["label"],
                                value=str(profile["default"]),
                                help=profile["description"],
                                key=f"single_{field_name}",
                            )
                        elif profile["kind"] == "numeric":
                            raw_values[field_name] = st.text_input(
                                profile["label"],
                                value=format_number(profile["default"]),
                                help=f"{profile['description']} Leave blank to keep this field missing.",
                                key=f"single_{field_name}",
                            )
                        else:
                            options = ["(Missing)"] + profile["options"]
                            default_value = profile["default"] if profile["default"] in profile["options"] else "(Missing)"
                            raw_values[field_name] = st.selectbox(
                                profile["label"],
                                options=options,
                                index=options.index(default_value),
                                help=profile["description"],
                                key=f"single_{field_name}",
                            )

        submitted = st.form_submit_button("Score this SME", use_container_width=True)

    if not submitted:
        return None

    record: dict[str, Any] = {}
    validation_errors: list[str] = []
    for field_name in RAW_INPUT_COLUMNS:
        profile = field_profiles[field_name]
        raw_value = raw_values[field_name]

        if profile["kind"] == "text":
            record[field_name] = raw_value.strip() or "APP_RECORD_001"
            continue

        if profile["kind"] == "numeric":
            parsed_value, error_message = parse_optional_numeric(field_name, raw_value)
            if error_message:
                validation_errors.append(error_message)
            record[field_name] = parsed_value
            continue

        record[field_name] = None if raw_value == "(Missing)" else raw_value

    if validation_errors:
        for message in validation_errors:
            st.error(message)
        return None

    return record


def resolve_artifact(uploaded_artifact: Any) -> tuple[dict[str, Any] | None, str | None]:
    if uploaded_artifact is not None:
        artifact_payload = load_artifact_from_bytes(uploaded_artifact.getvalue())
        return artifact_payload, "Uploaded artifact"

    if DEFAULT_ARTIFACT_PATH.exists():
        artifact_payload = load_artifact_from_path(str(DEFAULT_ARTIFACT_PATH))
        return artifact_payload, f"Local artifact: `{DEFAULT_ARTIFACT_PATH}`"

    return None, None


def prepare_batch_frame(uploaded_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], bool]:
    prepared_df = uploaded_df.copy()
    generated_ids = False

    if ID_COLUMN not in prepared_df.columns:
        generated_ids = True
        prepared_df.insert(0, ID_COLUMN, [f"APP_BATCH_{idx:05d}" for idx in range(1, len(prepared_df) + 1)])

    missing_columns = [column for column in RAW_INPUT_COLUMNS if column not in prepared_df.columns]
    for column in missing_columns:
        prepared_df[column] = pd.NA

    extra_columns = [column for column in prepared_df.columns if column not in RAW_INPUT_COLUMNS]
    for column in NUMERIC_FIELDS:
        prepared_df[column] = pd.to_numeric(prepared_df[column], errors="coerce")

    ordered_columns = RAW_INPUT_COLUMNS + [column for column in prepared_df.columns if column not in RAW_INPUT_COLUMNS]
    prepared_df = prepared_df[ordered_columns]
    return prepared_df, missing_columns, extra_columns, generated_ids


def render_prediction_summary(prediction_row: pd.Series) -> None:
    probability_frame = (
        prediction_row.filter(like="prob_")
        .rename(lambda column_name: column_name.replace("prob_", ""))
        .sort_values(ascending=False)
        .rename_axis("Class")
        .reset_index(name="Probability")
    )
    confidence = float(probability_frame["Probability"].max()) * 100

    metric_col, detail_col = st.columns([1.1, 1.3])
    with metric_col:
        st.metric("Predicted financial health", str(prediction_row["Target"]))
        st.metric("Top-class confidence", f"{confidence:.1f}%")
    with detail_col:
        st.dataframe(
            probability_frame.assign(
                Probability=lambda frame: frame["Probability"].map(lambda value: f"{value:.2%}")
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.bar_chart(probability_frame.set_index("Class"))


def render_story_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="story-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_eda_story() -> None:
    eda_summary = load_eda_summary()
    report = load_classification_report()
    model_comparison = load_model_comparison()
    subgroup_analysis = load_subgroup_analysis()
    eda_report_text = load_eda_report_text()

    st.subheader("Data story and model narrative")
    st.markdown(
        """
        <div class="story-band">
            <h3>What this portfolio looks like</h3>
            <p>
                This dataset is not just about raw revenue. It captures resilience, access to finance,
                business maturity, owner sentiment, and informal coping mechanisms across four countries.
                The strongest story in the data is that context matters: country patterns, missingness, and
                access signals are all carrying real predictive information.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if eda_summary:
        rows = eda_summary.get("shape", {}).get("rows", 0)
        columns = eda_summary.get("shape", {}).get("columns", 0)
        class_distribution = eda_summary.get("target_distribution", {})
        high_share = float(class_distribution.get("High", 0.0)) * 100
        low_share = float(class_distribution.get("Low", 0.0)) * 100
        country_distribution = eda_summary.get("country_distribution", {})
        top_country = max(country_distribution, key=country_distribution.get) if country_distribution else "N/A"

        metric_cols = st.columns(4)
        metric_cols[0].metric("Training rows", f"{rows:,}")
        metric_cols[1].metric("Input fields", str(columns))
        metric_cols[2].metric("Low-class share", f"{low_share:.1f}%")
        metric_cols[3].metric("High-class share", f"{high_share:.1f}%")

        story_cols = st.columns(3)
        with story_cols[0]:
            render_story_card(
                "Class imbalance defines the challenge",
                f"`Low` dominates the portfolio at {low_share:.1f}%, while `High` is only {high_share:.1f}%. "
                "That means a good model cannot just chase overall accuracy; it has to preserve sensitivity to the rare high-health segment.",
            )
        with story_cols[1]:
            render_story_card(
                "Country is a real business context signal",
                f"The largest country slice is {top_country.title()}, and the target mix varies materially by country. "
                "That tells us financial health is being shaped by local operating environments, not just firm-level variables.",
            )
        with story_cols[2]:
            top_missing_features = eda_summary.get("top_missing_features", {})
            if top_missing_features:
                first_feature, first_share = next(iter(top_missing_features.items()))
                body = (
                    f"`{first_feature}` is missing in about {float(first_share) * 100:.1f}% of records. "
                    "In this project, missingness is treated as business signal rather than simple noise, especially around finance access and informal borrowing."
                )
            else:
                body = "Missingness is widespread across access-to-finance fields, so blank values themselves become part of the business story."
            render_story_card("Missing data is part of the story", body)

    image_specs = [
        ("target_distribution.png", "Target distribution", "The portfolio is heavily weighted toward `Low`, making rare-class detection one of the central modeling problems."),
        ("country_target_share.png", "Country-level target mix", "Country effects are strong enough to justify explicit location-aware modeling rather than assuming a single uniform business environment."),
        ("missingness_top20.png", "Top missingness features", "The sparsest variables cluster around finance access, insurance, and informal borrowing, which suggests that data availability itself reflects business formalization."),
        ("monetary_boxplots.png", "Monetary features by class", "Revenue and expense signals are highly skewed, which is why the training pipeline leans on log transforms and robust ratio features."),
    ]
    for image_name, heading, caption in image_specs:
        image_path = EDA_IMAGE_DIR / image_name
        if image_path.exists():
            st.markdown(f"**{heading}**")
            st.image(str(image_path), use_container_width=True)
            st.caption(caption)

    if not model_comparison.empty:
        st.markdown("**Model leaderboard**")
        leaderboard = model_comparison[
            ["model", "weighted_f1", "macro_f1", "high_f1", "medium_f1", "low_f1"]
        ].copy()
        leaderboard = leaderboard.head(8)
        for metric_name in ["weighted_f1", "macro_f1", "high_f1", "medium_f1", "low_f1"]:
            leaderboard[metric_name] = leaderboard[metric_name].map(lambda value: f"{value:.3f}")
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

        best_row = model_comparison.iloc[0]
        st.markdown(
            f"""
            <div class="story-band">
                <h3>Why the ensemble won</h3>
                <p>
                    The selected model is <strong>{best_row['model']}</strong>, with weighted F1 of
                    <strong>{best_row['weighted_f1']:.3f}</strong> and macro F1 of
                    <strong>{best_row['macro_f1']:.3f}</strong>. The ensemble wins because it balances strong `Low`
                    performance with meaningfully better rare-class handling than simpler baselines.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if report:
        performance_cols = st.columns(3)
        for idx, label in enumerate(["Low", "Medium", "High"]):
            label_block = report.get(label, {})
            performance_cols[idx].metric(
                f"{label} F1",
                f"{float(label_block.get('f1-score', 0.0)):.3f}",
                delta=f"Recall {float(label_block.get('recall', 0.0)):.3f}",
            )

    if not subgroup_analysis.empty:
        st.markdown("**Where the model is strongest and weakest**")
        country_slice = (
            subgroup_analysis[subgroup_analysis["segment"] == "country"]
            .sort_values("macro_f1", ascending=False)
            .reset_index(drop=True)
        )
        if not country_slice.empty:
            weakest = country_slice.iloc[-1]
            strongest = country_slice.iloc[0]
            subgroup_cols = st.columns(2)
            with subgroup_cols[0]:
                st.dataframe(country_slice, use_container_width=True, hide_index=True)
            with subgroup_cols[1]:
                render_story_card(
                    "Most reliable market",
                    f"{strongest['group'].title()} leads with macro F1 of {strongest['macro_f1']:.3f}. "
                    "That suggests its pattern of business health is more separable under the current feature set.",
                )
                render_story_card(
                    "Primary risk pocket",
                    f"{weakest['group'].title()} is the weakest slice, with macro F1 of {weakest['macro_f1']:.3f}. "
                    "This is the clearest place to focus the next round of feature work, calibration, or country-specific modeling.",
                )

    if eda_report_text:
        with st.expander("EDA report notes"):
            st.markdown(eda_report_text)


def main() -> None:
    st.set_page_config(
        page_title="SME Financial Health Studio",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    field_profiles = build_field_profiles()
    run_summary = load_run_summary()

    eda_summary = load_eda_summary()
    class_distribution = eda_summary.get("target_distribution", {}) if eda_summary else {}
    high_share = float(class_distribution.get("High", 0.0)) * 100
    low_share = float(class_distribution.get("Low", 0.0)) * 100

    st.markdown(
        """
        <section class="hero-card">
            <h1>SME Financial Health Studio</h1>
            <p>
                Score individual businesses, explore the portfolio story, and review model evidence from the same
                SME Financial Health Index pipeline used for training.
            </p>
            <div class="hero-pills">
                <span class="hero-pill">Single-record scoring</span>
                <span class="hero-pill">Batch CSV scoring</span>
                <span class="hero-pill">EDA storytelling</span>
                <span class="hero-pill">Probability outputs</span>
                <span class="hero-pill">Country-level insights</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    hero_metrics = st.columns(4)
    hero_metrics[0].metric("Selected model", run_summary.get("selected_model", "unknown") if run_summary else "unknown")
    hero_metrics[1].metric("Best single model", run_summary.get("best_single_model", "unknown") if run_summary else "unknown")
    hero_metrics[2].metric("Low-class share", f"{low_share:.1f}%")
    hero_metrics[3].metric("High-class share", f"{high_share:.1f}%")

    st.sidebar.header("App controls")
    uploaded_artifact = st.sidebar.file_uploader(
        "Optional model artifact upload",
        type=["joblib"],
        help="Use this if `artifacts/trained_pipeline.joblib` is not available locally.",
    )

    artifact_payload, artifact_source = resolve_artifact(uploaded_artifact)
    if artifact_payload is None:
        st.error(
            "No trained model artifact was found. Run `python train.py` to create "
            "`artifacts/trained_pipeline.joblib`, or upload a `.joblib` artifact in the sidebar."
        )
        st.stop()

    st.sidebar.success("Prediction engine ready")
    st.sidebar.caption(artifact_source)
    if run_summary:
        st.sidebar.markdown(f"Selected model: `{run_summary.get('selected_model', 'unknown')}`")
        st.sidebar.markdown(f"Best single model: `{run_summary.get('best_single_model', 'unknown')}`")

    single_tab, batch_tab, story_tab, about_tab = st.tabs(
        ["Single SME", "Batch scoring", "Data story", "About the app"]
    )

    with single_tab:
        st.subheader("Score one SME at a time")
        single_record = render_single_record_form(field_profiles)
        if single_record is not None:
            input_df = pd.DataFrame([single_record], columns=RAW_INPUT_COLUMNS)
            prediction_frame = predict_dataframe(input_df, artifact_payload)
            st.success("Prediction complete")
            render_prediction_summary(prediction_frame.iloc[0])
            with st.expander("Preview the submitted record"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

    with batch_tab:
        st.subheader("Score a CSV batch")
        st.markdown(
            "Upload any CSV that contains at least some of the expected SME input columns. "
            "Missing columns are auto-filled as blanks, and IDs are generated if needed."
        )

        template_col, example_col = st.columns(2)
        with template_col:
            st.download_button(
                "Download blank template",
                data=dataframe_to_csv_bytes(build_blank_template()),
                file_name="sme_input_template_blank.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with example_col:
            st.download_button(
                "Download example template",
                data=dataframe_to_csv_bytes(build_example_template(field_profiles)),
                file_name="sme_input_template_example.csv",
                mime="text/csv",
                use_container_width=True,
            )

        uploaded_csv = st.file_uploader("Upload input CSV", type=["csv"], key="batch_uploader")
        use_local_test = st.button(
            "Load bundled local test file",
            disabled=not TEST_PATH.exists(),
            use_container_width=True,
        )

        batch_source_df: pd.DataFrame | None = None
        batch_source_name = ""
        if uploaded_csv is not None:
            batch_source_df = pd.read_csv(uploaded_csv)
            batch_source_name = uploaded_csv.name
        elif use_local_test and TEST_PATH.exists():
            batch_source_df = pd.read_csv(TEST_PATH)
            batch_source_name = "data/raw/Test.csv"

        if batch_source_df is not None:
            prepared_df, missing_columns, extra_columns, generated_ids = prepare_batch_frame(batch_source_df)

            info_messages = []
            if generated_ids:
                info_messages.append("Generated IDs because the upload did not include an `ID` column.")
            if missing_columns:
                info_messages.append(
                    f"Auto-filled missing columns with blanks: {', '.join(missing_columns)}."
                )
            if extra_columns:
                info_messages.append(
                    f"Kept extra columns in the scored export: {', '.join(extra_columns)}."
                )
            for message in info_messages:
                st.info(message)

            st.caption(f"Current batch source: `{batch_source_name}`")
            st.dataframe(prepared_df.head(25), use_container_width=True, hide_index=True)

            if st.button("Run batch scoring", use_container_width=True):
                scoring_input = prepared_df[RAW_INPUT_COLUMNS].copy()
                predictions = predict_dataframe(scoring_input, artifact_payload)
                batch_results = pd.concat(
                    [
                        prepared_df.reset_index(drop=True),
                        predictions.drop(columns=[ID_COLUMN]).reset_index(drop=True),
                    ],
                    axis=1,
                )
                summary = (
                    batch_results["Target"].value_counts().rename_axis("Predicted class").reset_index(name="Count")
                )

                st.success("Batch scoring complete")
                summary_col, download_col = st.columns([1.2, 1.0])
                with summary_col:
                    st.dataframe(summary, use_container_width=True, hide_index=True)
                    st.bar_chart(summary.set_index("Predicted class"))
                with download_col:
                    st.download_button(
                        "Download scored batch",
                        data=dataframe_to_csv_bytes(batch_results),
                        file_name="sme_scored_batch.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with st.expander("Preview scored rows", expanded=True):
                    st.dataframe(batch_results.head(50), use_container_width=True, hide_index=True)

    with story_tab:
        render_eda_story()

    with about_tab:
        st.subheader("How this frontend works")
        st.markdown(
            """
            - Uses the saved training artifact and the shared Python inference pipeline.
            - Supports both one-record interactive scoring and CSV batch scoring.
            - Returns the predicted class plus per-class probabilities.
            - Accepts partial batch uploads by auto-filling missing feature columns.
            """
        )
        if VARIABLE_DEFINITIONS_PATH.exists():
            with st.expander("Input dictionary", expanded=True):
                definitions_df = pd.read_csv(VARIABLE_DEFINITIONS_PATH)
                st.dataframe(definitions_df, use_container_width=True, hide_index=True)
        with st.expander("Expected raw input columns"):
            st.dataframe(
                pd.DataFrame({"column": RAW_INPUT_COLUMNS}),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
