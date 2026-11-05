"""
Model Architecture and Training Pipeline.
Supports Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and MLP.
Implements imbalance strategies: Class Weighting, SMOTE, and Threshold Optimization.
"""

import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.config import RANDOM_STATE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR
from src.evaluation import (
    evaluate_model_performance,
    find_optimal_threshold,
    plot_confusion_matrices,
    plot_roc_and_pr_curves,
)


def get_model_instances(scale_pos_weight: float = 1.0, random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Initializes standard benchmark models with balanced class penalties.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        ),
        "MLP Neural Net": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=150,
            early_stopping=True,
            random_state=random_state,
        ),
    }


def train_and_evaluate_all_models(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    X_val_proc: np.ndarray,
    y_val: np.ndarray,
    imbalance_strategy: str = "class_weight",  # "class_weight", "smote", "undersample"
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, float]]:
    """
    Trains all candidate models, optimizes probability thresholds, records comparison metrics,
    and identifies the top-performing model.
    """
    # 1. Apply sampling if requested (strictly on train set)
    X_train_fit = X_train_proc
    y_train_fit = y_train

    if imbalance_strategy == "smote":
        print("[Training] Applying SMOTE oversampling to training set...")
        sm = SMOTE(random_state=random_state)
        X_train_fit, y_train_fit = sm.fit_resample(X_train_proc, y_train)
    elif imbalance_strategy == "undersample":
        print("[Training] Applying Random Undersampling to training set...")
        rus = RandomUnderSampler(random_state=random_state)
        X_train_fit, y_train_fit = rus.fit_resample(X_train_proc, y_train)

    # Calculate negative/positive ratio for XGBoost
    n_neg = np.sum(y_train_fit == 0)
    n_pos = np.sum(y_train_fit == 1)
    pos_weight = float(n_neg / max(n_pos, 1))

    models = get_model_instances(scale_pos_weight=pos_weight, random_state=random_state)

    results = []
    fitted_models = {}
    best_thresholds = {}
    roc_pr_curves_data = {}
    confusion_matrix_data = {}

    print(f"\n{'='*75}")
    print(f"BENCHMARKING MODELS (Imbalance Strategy: {imbalance_strategy})")
    print(f"{'='*75}")

    for name, model in models.items():
        print(f"--> Training {name}...")
        start_time = time.time()
        model.fit(X_train_fit, y_train_fit)
        fit_time = time.time() - start_time

        fitted_models[name] = model

        # Predicted probabilities on validation set
        if hasattr(model, "predict_proba"):
            y_prob_val = model.predict_proba(X_val_proc)[:, 1]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(X_val_proc)
            y_prob_val = 1 / (1 + np.exp(-raw_scores))
        else:
            y_prob_val = model.predict(X_val_proc).astype(float)

        # Standard default evaluation at 0.50 threshold
        y_pred_default = (y_prob_val >= 0.50).astype(int)
        metrics_default = evaluate_model_performance(
            y_val, y_pred_default, y_prob_val, model_name=f"{name} (Thresh 0.50)", fit_time_sec=fit_time
        )
        results.append(metrics_default)

        # Threshold optimization for rare serious class
        opt_thresh, opt_f1 = find_optimal_threshold(y_val, y_prob_val, metric="f1")
        best_thresholds[name] = opt_thresh
        y_pred_tuned = (y_prob_val >= opt_thresh).astype(int)

        metrics_tuned = evaluate_model_performance(
            y_val, y_pred_tuned, y_prob_val, model_name=f"{name} (Tuned Thresh {opt_thresh})", fit_time_sec=fit_time
        )
        results.append(metrics_tuned)

        roc_pr_curves_data[name] = (y_val, y_prob_val)
        confusion_matrix_data[name] = (y_val, y_pred_tuned)

        print(f"    Macro F1: {metrics_tuned['Macro F1']:.3f} | Recall (Serious): {metrics_tuned['Recall (Serious)']:.3f} | PR-AUC: {metrics_tuned['PR-AUC']:.3f}")

    results_df = pd.DataFrame(results)

    # Save comparison table
    comparison_csv = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_csv, index=False)
    print(f"\n[Models] Saved comparison table to {comparison_csv}")

    # Plot and save ROC & PR curves
    plot_roc_and_pr_curves(roc_pr_curves_data, save_path=FIGURES_DIR / "roc_and_pr_curves.png")

    # Plot confusion matrices of top models
    top_models_cm = {k: confusion_matrix_data[k] for k in ["Random Forest", "XGBoost", "LightGBM"] if k in confusion_matrix_data}
    plot_confusion_matrices(top_models_cm, save_path=FIGURES_DIR / "confusion_matrices_top_models.png")

    return results_df, fitted_models, best_thresholds
