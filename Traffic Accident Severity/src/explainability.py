"""
Explainability and Risk Factor Analysis Module (Phase 8).
Implements:
1. SHAP (SHapley Additive exPlanations) summary & dependence plots.
2. Shallow Decision Tree rule extraction.
3. Odds Ratio (OR) and Relative Risk (RR) statistical analysis with plain-language conclusions.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sklearn.tree import export_text, DecisionTreeClassifier

from src.config import FIGURES_DIR, REPORTS_DIR


def generate_shap_analysis(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    max_display: int = 15,
    save_dir: Optional[Path] = FIGURES_DIR,
) -> Tuple[np.ndarray, shap.Explainer]:
    """
    Computes SHAP values using TreeExplainer and produces summary and bar plots.
    """
    print("[Explainability] Computing SHAP values on representative sample...")
    
    # Check model architecture for appropriate explainer
    try:
        if hasattr(model, "tree_") or hasattr(model, "estimators_") or "XGB" in type(model).__name__ or "LGBM" in type(model).__name__:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_sample[:100])
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, X_sample[:100])
            shap_values = explainer(X_sample).values
    except Exception as e:
        print(f"[Explainability] Standard explainer failed ({e}), using shap.Explainer...")
        explainer = shap.Explainer(model, X_sample[:50])
        shap_values = explainer(X_sample).values

    # Handle multi-class / binary output dimensions
    if isinstance(shap_values, list):
        # In binary classification, take SHAP values for class 1 (Serious)
        values_for_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif len(shap_values.shape) == 3:
        values_for_plot = shap_values[:, :, 1]
    else:
        values_for_plot = shap_values

    # 1. SHAP Beeswarm Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        values_for_plot,
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Beeswarm Plot: Feature Impact on Accident Severity", fontsize=12, pad=15)
    plt.tight_layout()
    summary_path = save_dir / "shap_summary_beeswarm.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Explainability] Saved SHAP beeswarm summary to {summary_path}")

    # 2. SHAP Bar Plot (Global Mean |SHAP|)
    plt.figure(figsize=(10, 7))
    mean_abs_shap = np.abs(values_for_plot).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:]
    
    plt.barh(
        range(len(top_indices)),
        mean_abs_shap[top_indices],
        color="#3b82f6",
        edgecolor="#1d4ed8",
    )
    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices], fontsize=10)
    plt.xlabel("Mean |SHAP Value| (Average Impact on Prediction)", fontsize=11)
    plt.title("Top Global Risk Predictors (Mean |SHAP|)", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    bar_path = save_dir / "shap_importance_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Explainability] Saved SHAP bar plot to {bar_path}")

    return values_for_plot, explainer


def extract_decision_tree_rules(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    max_depth: int = 3,
) -> str:
    """
    Fits an interpretable shallow decision tree and extracts human-readable decision rules.
    """
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=200,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X_train, y_train)
    rules_text = export_text(tree, feature_names=feature_names)
    
    rules_path = REPORTS_DIR / "decision_rules.txt"
    with open(rules_path, "w") as f:
        f.write("=== SHALLOW DECISION TREE EXTRACTED RULES (Serious vs Non-Serious) ===\n\n")
        f.write(rules_text)
    print(f"[Explainability] Saved interpretable rules to {rules_path}")

    return rules_text


def calculate_risk_odds_ratios(df: pd.DataFrame, target_col: str = "Is_Serious") -> pd.DataFrame:
    """
    Computes Odds Ratios (OR) and plain-language risk statements for key hazard conditions.
    """
    risk_factors = [
        ("Low Visibility (< 2 mi)", "Low_Visibility" if "Low_Visibility" in df.columns else None),
        ("Nighttime Driving (8 PM - 6 AM)", "Is_Night" if "Is_Night" in df.columns else None),
        ("Adverse Weather (Rain/Snow/Fog)", "Has_Adverse_Weather" if "Has_Adverse_Weather" in df.columns else None),
        ("Highway Junction", "Junction" if "Junction" in df.columns else None),
        ("Rush Hour Commute", "Is_Rush_Hour" if "Is_Rush_Hour" in df.columns else None),
        ("Weekend Travel", "Is_Weekend" if "Is_Weekend" in df.columns else None),
        ("Traffic Signal Present", "Traffic_Signal" if "Traffic_Signal" in df.columns else None),
    ]

    # Pre-calculate adverse weather if needed
    if "Weather_Condition_Group" in df.columns and "Has_Adverse_Weather" not in df.columns:
        df["Has_Adverse_Weather"] = df["Weather_Condition_Group"].isin(["Rain", "Snow/Ice", "Fog/Haze", "Severe_Storm"]).astype(int)

    results = []

    y = df[target_col].values

    for factor_label, col in risk_factors:
        if col is None or col not in df.columns:
            continue

        x = df[col].astype(int).values

        # Contingency table:
        # a = exposed & serious (x=1, y=1)
        # b = exposed & non-serious (x=1, y=0)
        # c = unexposed & serious (x=0, y=1)
        # d = unexposed & non-serious (x=0, y=0)
        a = np.sum((x == 1) & (y == 1))
        b = np.sum((x == 1) & (y == 0))
        c = np.sum((x == 0) & (y == 1))
        d = np.sum((x == 0) & (y == 0))

        # Avoid zero division with Haldane-Anscombe correction
        if b * c == 0:
            odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        else:
            odds_ratio = (a * d) / (b * c)

        # 95% Confidence Interval for Log Odds
        se_log_or = np.sqrt(1.0 / max(a, 1) + 1.0 / max(b, 1) + 1.0 / max(c, 1) + 1.0 / max(d, 1))
        ci_lower = np.exp(np.log(odds_ratio) - 1.96 * se_log_or)
        ci_upper = np.exp(np.log(odds_ratio) + 1.96 * se_log_or)

        # Serious rate when exposed vs unexposed
        rate_exposed = (a / max(a + b, 1)) * 100
        rate_unexposed = (c / max(c + d, 1)) * 100

        if odds_ratio >= 1.0:
            statement = f"Accidents with '{factor_label}' have {odds_ratio:.2f}x higher odds of being serious ({rate_exposed:.1f}% vs {rate_unexposed:.1f}%)."
        else:
            statement = f"Presence of '{factor_label}' is associated with lower odds of severity (OR = {odds_ratio:.2f}, {rate_exposed:.1f}% vs {rate_unexposed:.1f}%)."

        results.append({
            "Risk Factor": factor_label,
            "Odds Ratio (OR)": round(odds_ratio, 2),
            "95% CI Lower": round(ci_lower, 2),
            "95% CI Upper": round(ci_upper, 2),
            "Rate Exposed (%)": round(rate_exposed, 1),
            "Rate Baseline (%)": round(rate_unexposed, 1),
            "Plain-Language Finding": statement,
        })

    odds_df = pd.DataFrame(results)
    odds_csv = REPORTS_DIR / "risk_factor_odds_ratios.csv"
    odds_df.to_csv(odds_csv, index=False)
    print(f"[Explainability] Saved Risk Factor Odds Ratios to {odds_csv}")

    # Generate Markdown summary for final project presentation
    report_md = REPORTS_DIR / "risk_factors_findings.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Key Findings: Conditions Linked to Serious Road Traffic Accidents\n\n")
        f.write("This report details the statistical risk factors and machine learning explainability findings (Phase 8).\n\n")
        f.write("### 1. Statistical Odds Ratios & Relative Impact\n\n")
        f.write(odds_df[["Risk Factor", "Odds Ratio (OR)", "95% CI Lower", "95% CI Upper", "Rate Exposed (%)", "Rate Baseline (%)"]].to_markdown(index=False))
        f.write("\n\n### 2. Plain-Language Practical Takeaways\n\n")
        for stmt in odds_df["Plain-Language Finding"]:
            f.write(f"- {stmt}\n")

    return odds_df
