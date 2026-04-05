from __future__ import annotations

import os
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid")

MIN_BRAND_COUNT = 10
MAJOR_BRAND_COUNT = 50
ALLOWED_COMPANIES = ["ASUS", "Lenovo", "HP", "DELL", "Acer"]
COMPANY_NAME_MAP = {
    "asus": "ASUS",
    "lenovo": "Lenovo",
    "hp": "HP",
    "hewlett packard": "HP",
    "dell": "DELL",
    "acer": "Acer",
}
CORE_ATTRIBUTES_OVERALL = [
    "Company_Bucket_10",
    "Price_Bucket",
    "Processor_Tier",
    "RAM_Bucket",
    "Storage_Bucket",
    "Warranty_Bucket",
]
CORE_ATTRIBUTES_WITHIN_BRAND = [
    "Price_Bucket",
    "Processor_Tier",
    "RAM_Bucket",
    "Storage_Bucket",
    "Warranty_Bucket",
]
TARGET_COL = "Star_Rating_num"
ATTRIBUTE_LABEL_MAP = {
    "Company_Bucket_10": "Company",
    "Price_Bucket": "Price",
    "Processor_Tier": "Processor",
    "RAM_Bucket": "RAM",
    "Storage_Bucket": "Storage",
    "Warranty_Bucket": "Warranty",
}


def norm(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_missing(value) -> bool:
    return norm(value) in {"", "N/A", "NA", "None", "null", "NULL", "-"}


def parse_float(value):
    value = norm(value)
    if is_missing(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def parse_int(value):
    value = norm(value)
    if is_missing(value):
        return np.nan
    try:
        return int(float(value))
    except Exception:
        return np.nan


def parse_ram_gb(value):
    match = re.search(r"(\d+)\s*gb", norm(value).lower())
    return int(match.group(1)) if match else np.nan


def parse_storage_gb(value):
    value = norm(value)
    if is_missing(value):
        return np.nan
    try:
        numeric = float(value)
    except Exception:
        return np.nan
    return int(numeric * 1024) if numeric <= 10 else int(numeric)


def bucket_price(value) -> str:
    value = parse_int(value)
    if pd.isna(value):
        return "Missing"
    if value < 30000:
        return "Budget"
    if value < 60000:
        return "Mid-range"
    return "Premium"


def bucket_processor_family(value) -> str:
    text = norm(value).lower()
    if is_missing(text):
        return "Missing"
    checks = [
        ("celeron", "Intel Celeron"),
        ("pentium", "Intel Pentium"),
        ("core ultra 9", "Intel Core Ultra 9"),
        ("core ultra 7", "Intel Core Ultra 7"),
        ("core ultra 5", "Intel Core Ultra 5"),
        ("core i9", "Intel Core i9"),
        ("core i7", "Intel Core i7"),
        ("core i5", "Intel Core i5"),
        ("core i3", "Intel Core i3"),
        ("core 9", "Intel Core 9"),
        ("core 7", "Intel Core 7"),
        ("core 5", "Intel Core 5"),
        ("core 3", "Intel Core 3"),
        ("ryzen ai 9", "AMD Ryzen AI 9"),
        ("ryzen ai 7", "AMD Ryzen AI 7"),
        ("ryzen 9", "AMD Ryzen 9"),
        ("ryzen 7", "AMD Ryzen 7"),
        ("ryzen 5", "AMD Ryzen 5"),
        ("ryzen 3", "AMD Ryzen 3"),
        ("athlon", "AMD Athlon"),
        ("snapdragon", "Snapdragon"),
        ("mediatek", "MediaTek"),
    ]
    if "ultra" not in text:
        for key, label in checks:
            if key in text:
                return label
    for key, label in checks:
        if key in text:
            return label
    for apple in ["m4", "m3", "m2", "m1"]:
        if apple in text:
            return f"Apple {apple.upper()}"
    return "Other"


def bucket_processor_tier(value) -> str:
    family = bucket_processor_family(value)
    if family in {"Intel Celeron", "Intel Pentium", "AMD Athlon", "MediaTek"}:
        return "Entry"
    if family in {"Intel Core i3", "Intel Core 3", "AMD Ryzen 3", "Snapdragon"}:
        return "Mainstream"
    if family in {"Intel Core i5", "Intel Core 5", "AMD Ryzen 5"}:
        return "Upper mainstream"
    if family in {"Intel Core i7", "Intel Core 7", "AMD Ryzen 7", "Intel Core Ultra 5"}:
        return "Performance"
    if family in {
        "Intel Core i9",
        "Intel Core 9",
        "Intel Core Ultra 7",
        "Intel Core Ultra 9",
        "AMD Ryzen 9",
        "AMD Ryzen AI 7",
        "AMD Ryzen AI 9",
    }:
        return "Premium performance"
    if family.startswith("Apple "):
        return "Apple silicon"
    return "Other"


def bucket_ram(value) -> str:
    value = parse_ram_gb(value)
    if pd.isna(value):
        return "Missing"
    if value <= 8:
        return "8 GB or less"
    if value <= 16:
        return "16 GB"
    return "24 GB or more"


def bucket_storage(value) -> str:
    value = parse_storage_gb(value)
    if pd.isna(value):
        return "Missing"
    if value <= 256:
        return "256 GB or less"
    if value <= 512:
        return "512 GB"
    return "1 TB or more"


def bucket_warranty(value) -> str:
    value = parse_int(value)
    if pd.isna(value):
        return "Missing"
    if value <= 1:
        return "1 year"
    if value == 2:
        return "2 years"
    return "3+ years"


def canonical_company_name(value) -> str:
    key = norm(value).lower()
    return COMPANY_NAME_MAP.get(key, norm(value))


def get_ordered_categories(values, preferred):
    present = [item for item in preferred if item in values]
    remainder = [item for item in values if item not in preferred]
    return present + remainder


def prepare_attribute_categories(df_in: pd.DataFrame, attribute_cols: list[str]) -> pd.DataFrame:
    df_out = df_in.copy()
    preferred_orders = {
        "Company_Bucket_10": ["ASUS", "Lenovo", "HP", "DELL", "Acer"],
        "Price_Bucket": ["Budget", "Mid-range", "Premium"],
        "Processor_Tier": ["Entry", "Mainstream", "Upper mainstream", "Performance", "Premium performance", "Apple silicon", "Other"],
        "RAM_Bucket": ["8 GB or less", "16 GB", "24 GB or more"],
        "Storage_Bucket": ["256 GB or less", "512 GB", "1 TB or more"],
        "Warranty_Bucket": ["1 year", "2 years", "3+ years"],
    }
    for column in attribute_cols:
        values = df_out[column].dropna().astype(str).unique().tolist()
        ordered = get_ordered_categories(values, preferred_orders.get(column, []))
        df_out[column] = pd.Categorical(df_out[column], categories=ordered, ordered=True)
    return df_out


def build_design_matrix(df_in: pd.DataFrame, attribute_cols: list[str]) -> pd.DataFrame:
    design = pd.get_dummies(df_in[attribute_cols], drop_first=True, prefix_sep="::").astype(float)
    return sm.add_constant(design, has_constant="add").astype(float)


def fit_conjoint_like_model(df_in: pd.DataFrame, attribute_cols: list[str], target_col: str, test_size: float = 0.2, random_state: int = 42) -> dict:
    df_model = prepare_attribute_categories(df_in.copy(), attribute_cols)
    X = build_design_matrix(df_model, attribute_cols)
    y = pd.to_numeric(df_model[target_col], errors="coerce").astype(float)

    valid = y.notna() & X.notna().all(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]
    df_model = df_model.loc[valid].copy()

    model = sm.OLS(y, X).fit(cov_type="HC3")
    idx_train, idx_test = train_test_split(X.index, test_size=test_size, random_state=random_state)
    model_train = sm.OLS(y.loc[idx_train], X.loc[idx_train]).fit()
    preds_test = model_train.predict(X.loc[idx_test])

    holdout_rmse = float(np.sqrt(np.mean((y.loc[idx_test] - preds_test) ** 2)))
    holdout_mae = float(np.mean(np.abs(y.loc[idx_test] - preds_test)))
    denom = np.sum((y.loc[idx_test] - y.loc[idx_test].mean()) ** 2)
    holdout_r2 = float(1 - np.sum((y.loc[idx_test] - preds_test) ** 2) / denom) if denom > 0 else np.nan

    coef_df = pd.DataFrame({"term": model.params.index, "beta": model.params.values, "p_value": model.pvalues.values})
    coef_df = coef_df[coef_df["term"] != "const"].copy()
    coef_df[["attribute", "level"]] = coef_df["term"].str.split("::", n=1, expand=True)

    partworth_rows = []
    for attr in attribute_cols:
        levels = list(df_model[attr].cat.categories)
        baseline = levels[0]
        raw_utils = {baseline: 0.0}
        for _, row in coef_df.loc[coef_df["attribute"] == attr].iterrows():
            raw_utils[row["level"]] = row["beta"]
        for level in levels:
            partworth_rows.append(
                {
                    "attribute": attr,
                    "level": level,
                    "raw_partworth_vs_baseline": float(raw_utils.get(level, 0.0)),
                }
            )

    partworth_df = pd.DataFrame(partworth_rows)
    partworth_df["zero_centered_partworth"] = partworth_df.groupby("attribute")["raw_partworth_vs_baseline"].transform(
        lambda series: series - series.mean()
    )

    importance_df = (
        partworth_df.groupby("attribute")["zero_centered_partworth"]
        .agg(lambda series: float(series.max() - series.min()))
        .reset_index(name="utility_range")
    )
    total_range = importance_df["utility_range"].sum()
    importance_df["importance_pct"] = 100 * importance_df["utility_range"] / total_range if total_range else 0.0

    fit_stats = pd.DataFrame(
        [
            {
                "n_obs": len(y),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "holdout_r2": holdout_r2,
                "holdout_rmse": holdout_rmse,
                "holdout_mae": holdout_mae,
            }
        ]
    )

    return {
        "model": model,
        "X": X,
        "y": y,
        "coef_df": coef_df,
        "partworth_df": partworth_df,
        "importance_df": importance_df,
        "fit_stats": fit_stats,
    }


def prepare_input_dataframe(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    raw_df = pd.read_csv(csv_path)
    feature_df = raw_df.copy()

    required_cols = ["Company", "Price", "Processor", "RAM", "Storage", "Warranty", "Star_Rating"]
    missing_cols = [column for column in required_cols if column not in feature_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    feature_df["Company"] = feature_df["Company"].apply(canonical_company_name)
    feature_df = feature_df.loc[feature_df["Company"].isin(ALLOWED_COMPANIES)].copy()

    feature_df["Price_num"] = feature_df["Price"].apply(parse_int)
    feature_df["Star_Rating_num"] = feature_df["Star_Rating"].apply(parse_float)
    feature_df["RAM_GB"] = feature_df["RAM"].apply(parse_ram_gb)
    feature_df["Storage_GB"] = feature_df["Storage"].apply(parse_storage_gb)
    feature_df["Warranty_Years"] = feature_df["Warranty"].apply(parse_int)

    company_counts = feature_df["Company"].astype(str).str.strip().value_counts()
    brands_major = sorted(company_counts[company_counts >= MAJOR_BRAND_COUNT].index.tolist())

    feature_df["Company_Bucket_10"] = feature_df["Company"]
    feature_df["Price_Bucket"] = feature_df["Price"].apply(bucket_price)
    feature_df["Processor_Tier"] = feature_df["Processor"].apply(bucket_processor_tier)
    feature_df["RAM_Bucket"] = feature_df["RAM"].apply(bucket_ram)
    feature_df["Storage_Bucket"] = feature_df["Storage"].apply(bucket_storage)
    feature_df["Warranty_Bucket"] = feature_df["Warranty"].apply(bucket_warranty)

    overall_df = feature_df.loc[
        feature_df["Star_Rating_num"].notna()
        & feature_df["Price_Bucket"].ne("Missing")
        & feature_df["Processor_Tier"].ne("Missing")
        & feature_df["RAM_Bucket"].ne("Missing")
        & feature_df["Storage_Bucket"].ne("Missing")
        & feature_df["Warranty_Bucket"].ne("Missing")
    ].copy()

    return overall_df, brands_major


def save_barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    output_path: Path,
    color: str,
    rotate_xticks: bool = False,
    hue: str | None = None,
) -> None:
    plt.figure(figsize=(9, 4 if len(data) <= 8 else 6))
    sns.barplot(data=data, x=x, y=y, color=None if hue else color, hue=hue)
    plt.title(title)
    plt.xlabel("" if x == "level" else x.replace("_", " "))
    plt.ylabel("" if y == "attribute" else y.replace("_", " "))
    if rotate_xticks:
        plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def export_overall_outputs(overall_results: dict, overall_df: pd.DataFrame, output_dir: Path) -> None:
    overall_results["coef_df"].to_csv(output_dir / "overall_betas.csv", index=False)
    overall_results["partworth_df"].to_csv(output_dir / "overall_partworths.csv", index=False)
    overall_results["importance_df"].to_csv(output_dir / "overall_importance.csv", index=False)
    overall_results["fit_stats"].to_csv(output_dir / "overall_fit_stats.csv", index=False)

    save_barplot(
        overall_results["importance_df"],
        x="importance_pct",
        y="attribute",
        title="Overall Attribute Importance",
        output_path=output_dir / "overall_attribute_importance.png",
        color="#264653",
    )

    for attr in CORE_ATTRIBUTES_OVERALL:
        plot_df = overall_results["partworth_df"].loc[overall_results["partworth_df"]["attribute"] == attr].copy()
        save_barplot(
            plot_df,
            x="level",
            y="zero_centered_partworth",
            title=f"Overall Zero-Centered Part-Worths: {attr}",
            output_path=output_dir / f"overall_{attr.lower()}_partworths.png",
            color="#2a9d8f",
            rotate_xticks=True,
        )

    model = overall_results["model"]
    overall_perf_panel = pd.DataFrame(
        [
            {
                "n_obs": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "f_statistic": float(model.fvalue) if model.fvalue is not None else np.nan,
                "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
                "holdout_r2": float(overall_results["fit_stats"]["holdout_r2"].iat[0]),
                "holdout_rmse": float(overall_results["fit_stats"]["holdout_rmse"].iat[0]),
                "holdout_mae": float(overall_results["fit_stats"]["holdout_mae"].iat[0]),
            }
        ]
    )
    overall_perf_panel.to_csv(output_dir / "overall_regression_performance.csv", index=False)

    X_vif = overall_results["X"].drop(columns="const", errors="ignore").copy()
    if X_vif.shape[1] > 0:
        overall_vif_df = pd.DataFrame(
            {
                "term": X_vif.columns,
                "vif": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
            }
        )
        overall_vif_df[["attribute", "level"]] = overall_vif_df["term"].str.split("::", n=1, expand=True)
        overall_vif_summary = (
            overall_vif_df.groupby("attribute")["vif"]
            .agg(["max", "mean"])
            .reset_index()
            .rename(columns={"max": "max_vif", "mean": "mean_vif"})
        )
        overall_vif_df.to_csv(output_dir / "overall_vif_by_term.csv", index=False)
        overall_vif_summary.to_csv(output_dir / "overall_vif_by_attribute.csv", index=False)

    overall_level_support = []
    for attr in CORE_ATTRIBUTES_OVERALL:
        tmp = overall_df[attr].value_counts(dropna=False).rename_axis("level").reset_index(name="n_products")
        tmp.insert(0, "attribute", attr)
        overall_level_support.append(tmp)
    pd.concat(overall_level_support, ignore_index=True).to_csv(output_dir / "ppt_overall_level_support.csv", index=False)

    overall_partworths = overall_results["partworth_df"].copy()
    overall_importance = overall_results["importance_df"].copy()
    overall_fit_stats = overall_results["fit_stats"].copy()

    overall_best_levels = (
        overall_partworths.sort_values(["attribute", "zero_centered_partworth"], ascending=[True, False])
        .groupby("attribute")
        .first()
        .reset_index()[["attribute", "level", "raw_partworth_vs_baseline", "zero_centered_partworth"]]
        .rename(columns={"level": "best_level"})
    )
    overall_worst_levels = (
        overall_partworths.sort_values(["attribute", "zero_centered_partworth"], ascending=[True, True])
        .groupby("attribute")
        .first()
        .reset_index()[["attribute", "level", "raw_partworth_vs_baseline", "zero_centered_partworth"]]
        .rename(columns={"level": "worst_level"})
    )
    overall_best_worst = overall_best_levels.merge(
        overall_worst_levels[["attribute", "worst_level", "raw_partworth_vs_baseline", "zero_centered_partworth"]],
        on="attribute",
        suffixes=("_best", "_worst"),
    )

    overall_fit_stats.to_csv(output_dir / "ppt_overall_fit_summary.csv", index=False)
    overall_importance.to_csv(output_dir / "ppt_overall_attribute_importance.csv", index=False)
    overall_best_worst.to_csv(output_dir / "ppt_overall_best_worst_levels.csv", index=False)
    overall_partworths.to_csv(output_dir / "ppt_overall_partworths_full.csv", index=False)

    tab2_overall = overall_importance.copy()
    tab2_overall["scope"] = "Overall"
    tab2_overall["attribute"] = tab2_overall["attribute"].map(ATTRIBUTE_LABEL_MAP)
    tab2_overall = tab2_overall[["scope", "attribute", "utility_range", "importance_pct"]]

    tab3_overall = overall_partworths.copy()
    tab3_overall["scope"] = "Overall"
    tab3_overall["attribute"] = tab3_overall["attribute"].map(ATTRIBUTE_LABEL_MAP)
    tab3_overall = tab3_overall[["scope", "attribute", "level", "zero_centered_partworth"]]

    tab2_overall.to_csv(output_dir / "tab_2.csv", index=False)
    tab3_overall.to_csv(output_dir / "tab_3.csv", index=False)


def export_brand_outputs(brand_results: dict[str, dict], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows = []
    importance_rows = []
    partworth_rows = []

    for brand, result in brand_results.items():
        fit_row = result["fit_stats"].copy()
        fit_row.insert(0, "brand", brand)
        fit_rows.append(fit_row)
        importance_rows.append(result["importance_df"].assign(brand=brand))
        partworth_rows.append(result["partworth_df"].assign(brand=brand))

        result["coef_df"].to_csv(output_dir / f"{brand.lower()}_betas.csv", index=False)
        result["partworth_df"].to_csv(output_dir / f"{brand.lower()}_partworths.csv", index=False)
        result["importance_df"].to_csv(output_dir / f"{brand.lower()}_importance.csv", index=False)

        save_barplot(
            result["importance_df"],
            x="importance_pct",
            y="attribute",
            title=f"{brand}: Attribute Importance",
            output_path=output_dir / f"{brand.lower()}_importance.png",
            color="#457b9d",
        )
        for attr in CORE_ATTRIBUTES_WITHIN_BRAND:
            plot_df = result["partworth_df"].loc[result["partworth_df"]["attribute"] == attr].copy()
            save_barplot(
                plot_df,
                x="level",
                y="zero_centered_partworth",
                title=f"{brand}: Zero-Centered Part-Worths - {attr}",
                output_path=output_dir / f"{brand.lower()}_{attr.lower()}_partworths.png",
                color="#e76f51",
                rotate_xticks=True,
            )

    brand_fit_summary = pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame()
    brand_importance_compare = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()
    brand_partworths_compare = pd.concat(partworth_rows, ignore_index=True) if partworth_rows else pd.DataFrame()

    if len(brand_fit_summary):
        brand_fit_summary.to_csv(output_dir / "brand_fit_summary.csv", index=False)
    if len(brand_importance_compare):
        brand_importance_compare.to_csv(output_dir / "brand_importance_compare.csv", index=False)
        save_barplot(
            brand_importance_compare,
            x="importance_pct",
            y="attribute",
            title="Attribute Importance by Brand",
            output_path=output_dir / "brand_importance_compare.png",
            color="#1d3557",
            hue="brand",
        )
    if len(brand_partworths_compare):
        brand_partworths_compare.to_csv(output_dir / "brand_partworths_compare.csv", index=False)

        tab2_brand = brand_importance_compare.rename(columns={"brand": "scope"}).copy()
        tab2_brand["attribute"] = tab2_brand["attribute"].map(ATTRIBUTE_LABEL_MAP)
        tab2_brand = tab2_brand[["scope", "attribute", "utility_range", "importance_pct"]]

        tab3_brand = brand_partworths_compare.rename(columns={"brand": "scope"}).copy()
        tab3_brand["attribute"] = tab3_brand["attribute"].map(ATTRIBUTE_LABEL_MAP)
        tab3_brand = tab3_brand[["scope", "attribute", "level", "zero_centered_partworth"]]

        return tab2_brand, tab3_brand

    return pd.DataFrame(), pd.DataFrame()


def run_conjoint(csv_path: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_df, brands_major = prepare_input_dataframe(csv_path)
    overall_results = fit_conjoint_like_model(overall_df, CORE_ATTRIBUTES_OVERALL, TARGET_COL)
    export_overall_outputs(overall_results, overall_df, output_dir)

    brand_results = {}
    for brand in brands_major:
        brand_df = overall_df.loc[overall_df["Company"] == brand].copy()
        if len(brand_df) < 30:
            continue
        brand_results[brand] = fit_conjoint_like_model(brand_df, CORE_ATTRIBUTES_WITHIN_BRAND, TARGET_COL)

    tab2_brand, tab3_brand = export_brand_outputs(brand_results, output_dir)
    if len(tab2_brand):
        tab2_overall = pd.read_csv(output_dir / "tab_2.csv")
        pd.concat([tab2_overall, tab2_brand], ignore_index=True).to_csv(output_dir / "tab_2.csv", index=False)
    if len(tab3_brand):
        tab3_overall = pd.read_csv(output_dir / "tab_3.csv")
        pd.concat([tab3_overall, tab3_brand], ignore_index=True).to_csv(output_dir / "tab_3.csv", index=False)

    return {
        "overall_importance": overall_results["importance_df"],
        "overall_partworths": overall_results["partworth_df"],
    }
