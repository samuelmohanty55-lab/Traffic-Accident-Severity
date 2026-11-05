"""
Master Training and Benchmarking Pipeline.
Executes end-to-end: Data Loading -> Stratified Split -> Pipeline Preprocessing ->
Model Comparison -> Validation Threshold Optimization -> Test Set Final Evaluation ->
SHAP & Odds Ratio Risk Factor Analysis -> Model Artifact Serialization.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    BINARY_TARGET_COLUMN,
)
from src.data_loader import load_dataset
from src.preprocessor import (
    prepare_raw_data,
    get_feature_lists,
    build_preprocessor_pipeline,
    stratified_split,
    get_feature_names_from_preprocessor,
)
from src.models import train_and_evaluate_all_models
from src.evaluation import evaluate_model_performance
from src.explainability import (
    generate_shap_analysis,
    extract_decision_tree_rules,
    calculate_risk_odds_ratios,
)


def run_pipeline(sample_size: int = 30000, imbalance_strategy: str = "class_weight"):
    print("=" * 80)
    print("STARTING ROAD TRAFFIC ACCIDENT SEVERITY ML PIPELINE")
    print("=" * 80)

    # Step 1: Load Data
    raw_df = load_dataset(sample_size=sample_size)
    print(f"[Pipeline] Loaded raw records: {raw_df.shape}")

    # Step 2: Prepare & Clean Data (Drop leakage & impute/derive features)
    df = prepare_raw_data(raw_df)
    print(f"[Pipeline] Processed dataset shape: {df.shape}")
    print(f"[Pipeline] Serious Accidents prevalence: {df[BINARY_TARGET_COLUMN].mean()*100:.2f}%")

    # Step 3: Stratified Split (70/15/15)
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = stratified_split(
        df, target_col=BINARY_TARGET_COLUMN, random_state=RANDOM_STATE
    )

    # Step 4: Fit Preprocessing Pipeline strictly on Train Set
    num_cols, cat_cols, road_cols = get_feature_lists(X_train_df)
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols, road_cols, scale_numeric=True)

    print(f"[Pipeline] Features - Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}, Road Flags: {len(road_cols)}")
    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_val_proc = preprocessor.transform(X_val_df)
    X_test_proc = preprocessor.transform(X_test_df)

    feature_names = get_feature_names_from_preprocessor(preprocessor)
    print(f"[Pipeline] Transformed feature space dimensionality: {len(feature_names)}")

    # Step 5: Model Training & Validation Benchmark
    results_df, fitted_models, best_thresholds = train_and_evaluate_all_models(
        X_train_proc,
        y_train.values,
        X_val_proc,
        y_val.values,
        imbalance_strategy=imbalance_strategy,
    )

    # Step 6: Identify Top Model (based on Macro F1 on validation set)
    # Filter tuned models
    tuned_results = results_df[results_df["Model"].str.contains("Tuned Thresh")].copy()
    best_row = tuned_results.sort_values(by="Macro F1", ascending=False).iloc[0]
    best_model_name = best_row["Model"].split(" (Tuned")[0]
    best_threshold = best_thresholds[best_model_name]
    best_model = fitted_models[best_model_name]

    print("\n" + "=" * 80)
    print(f"SELECTED CHAMPION MODEL: {best_model_name}")
    print(f"Optimal Decision Threshold: {best_threshold:.3f}")
    print("=" * 80)

    # Step 7: Final Single Evaluation on Held-Out Test Set (Phase 9)
    print("\n[Phase 9] Evaluating Champion Model on Held-Out Test Set...")
    if hasattr(best_model, "predict_proba"):
        y_test_prob = best_model.predict_proba(X_test_proc)[:, 1]
    else:
        y_test_prob = best_model.predict(X_test_proc).astype(float)

    y_test_pred = (y_test_prob >= best_threshold).astype(int)
    test_metrics = evaluate_model_performance(
        y_test.values,
        y_test_pred,
        y_test_prob,
        model_name=f"FINAL TEST SET: {best_model_name} (Thresh {best_threshold})",
    )

    print("\nFINAL TEST PERFORMANCE:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:24}: {v:.4f}")
        else:
            print(f"  {k:24}: {v}")

    # Append test result to comparison CSV
    test_df = pd.DataFrame([test_metrics])
    test_csv = REPORTS_DIR / "final_test_evaluation.csv"
    test_df.to_csv(test_csv, index=False)

    # Step 8: Phase 8 - Explainability & Risk Factor Identification
    print("\n" + "=" * 80)
    print("PHASE 8: RISK FACTOR IDENTIFICATION & EXPLAINABILITY (SHAP & STATISTICAL ODDS)")
    print("=" * 80)

    # Compute Odds Ratios on dataset
    calculate_risk_odds_ratios(df, target_col=BINARY_TARGET_COLUMN)

    # Extract human readable Decision Tree rules
    extract_decision_tree_rules(X_train_proc, y_train.values, feature_names=feature_names, max_depth=3)

    # Generate SHAP summary plots on sample
    shap_sample_size = min(600, len(X_test_proc))
    shap_sample_idx = np.random.choice(len(X_test_proc), size=shap_sample_size, replace=False)
    generate_shap_analysis(
        best_model,
        X_test_proc[shap_sample_idx],
        feature_names=feature_names,
        max_display=15,
        save_dir=FIGURES_DIR,
    )

    # Step 9: Save Serialized Model Pipeline Artifacts
    full_pipeline = {
        "preprocessor": preprocessor,
        "model": best_model,
        "model_name": best_model_name,
        "optimal_threshold": best_threshold,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "road_cols": road_cols,
        "feature_names": feature_names,
    }
    model_artifact_path = MODELS_DIR / "best_model_pipeline.joblib"
    joblib.dump(full_pipeline, model_artifact_path)
    print(f"\n[Artifacts] Saved full inference pipeline to {model_artifact_path}")

    # Save metadata JSON for Streamlit app
    metadata = {
        "best_model_name": best_model_name,
        "optimal_threshold": float(best_threshold),
        "test_macro_f1": float(test_metrics["Macro F1"]),
        "test_recall_serious": float(test_metrics["Recall (Serious)"]),
        "test_precision_serious": float(test_metrics["Precision (Serious)"]),
        "test_roc_auc": float(test_metrics["ROC-AUC"]),
        "test_pr_auc": float(test_metrics["PR-AUC"]),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "road_flag_features": road_cols,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[Pipeline Complete] All phases executed successfully!")
    return full_pipeline, results_df, test_metrics


if __name__ == "__main__":
    run_pipeline(sample_size=30000, imbalance_strategy="class_weight")
