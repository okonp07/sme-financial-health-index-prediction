from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import EDA_DIR
from src.data.loaders import load_train_data, load_variable_definitions
from src.features.engineering import MONETARY_COLUMNS

sns.set_theme(style="whitegrid")


def save_target_distribution(train_df: pd.DataFrame) -> None:
    counts = train_df["Target"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    plt.title("Target Class Distribution")
    plt.xlabel("Financial Health Index")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "target_distribution.png", dpi=200)
    plt.close()


def save_country_target_heatmap(train_df: pd.DataFrame) -> None:
    share = pd.crosstab(train_df["country"], train_df["Target"], normalize="index")
    plt.figure(figsize=(8, 5))
    sns.heatmap(share, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Country-Level Target Mix")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "country_target_share.png", dpi=200)
    plt.close()


def save_numeric_correlations(train_df: pd.DataFrame) -> None:
    numeric_df = train_df.select_dtypes(include="number")
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "numeric_correlation.png", dpi=200)
    plt.close()


def save_monetary_boxplots(train_df: pd.DataFrame) -> None:
    melted = train_df[MONETARY_COLUMNS + ["Target"]].melt(
        id_vars="Target", var_name="feature", value_name="value"
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="feature", y="value", hue="Target", showfliers=False)
    plt.yscale("log")
    plt.title("Monetary Features by Target (log scale)")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "monetary_boxplots.png", dpi=200)
    plt.close()


def save_missingness(train_df: pd.DataFrame) -> None:
    missing = train_df.isna().mean().sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=missing.values, y=missing.index, hue=missing.index, palette="magma", legend=False)
    plt.title("Top 20 Missingness Rates")
    plt.xlabel("Missing Share")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "missingness_top20.png", dpi=200)
    plt.close()


def write_summary(train_df: pd.DataFrame, definitions_df: pd.DataFrame) -> None:
    summary = {
        "shape": {"rows": int(train_df.shape[0]), "columns": int(train_df.shape[1])},
        "target_distribution": train_df["Target"].value_counts(normalize=True).round(4).to_dict(),
        "country_distribution": train_df["country"].value_counts(normalize=True).round(4).to_dict(),
        "top_missing_features": train_df.isna().mean().sort_values(ascending=False).head(15).round(4).to_dict(),
        "duplicate_ids": int(train_df["ID"].duplicated().sum()),
        "duplicate_rows": int(train_df.duplicated().sum()),
        "variable_definitions_loaded": int(definitions_df.shape[0]),
    }
    (EDA_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    report_lines = [
        "# SME Financial Health EDA",
        "",
        f"- Rows: {train_df.shape[0]}",
        f"- Columns: {train_df.shape[1]}",
        f"- Duplicate IDs: {summary['duplicate_ids']}",
        f"- Duplicate rows: {summary['duplicate_rows']}",
        "",
        "## Target distribution",
    ]
    for label, share in summary["target_distribution"].items():
        report_lines.append(f"- {label}: {share:.2%}")

    report_lines.extend(["", "## Highest missingness features"])
    for feature, share in summary["top_missing_features"].items():
        report_lines.append(f"- {feature}: {share:.2%}")

    report_lines.extend(["", "## Data notes"])
    report_lines.append("- Target is heavily imbalanced, with `High` as the rarest class.")
    report_lines.append("- Country patterns are material, especially Malawi and Eswatini.")
    report_lines.append("- Monetary variables are highly skewed and need robust transformations.")
    report_lines.append("- Many access-to-finance fields contain meaningful missing/unknown states.")
    (EDA_DIR / "eda_report.md").write_text("\n".join(report_lines))


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_train_data()
    definitions_df = load_variable_definitions()

    save_target_distribution(train_df)
    save_country_target_heatmap(train_df)
    save_numeric_correlations(train_df)
    save_monetary_boxplots(train_df)
    save_missingness(train_df)
    write_summary(train_df, definitions_df)


if __name__ == "__main__":
    main()
