"""
Model Evaluation and Metrics Module.
Calculates classification metrics (Precision, Recall, F1, Macro F1, ROC-AUC, PR-AUC),
plots confusion matrices, ROC curves, PR curves, and handles probability threshold tuning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

from src.config import FIGURES_DIR


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model",
    fit_time_sec: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics with special emphasis on the serious minority class.
    """
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Serious)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Serious)": recall_score(y_true, y_pred, zero_division=0),
        "F1 (Serious)": f1_score(y_true, y_pred, zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Fit Time (s)": round(fit_time_sec, 3),
    }

    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["ROC-AUC"] = np.nan

        try:
            metrics["PR-AUC"] = average_precision_score(y_true, y_prob)
        except Exception:
            metrics["PR-AUC"] = np.nan
    else:
        metrics["ROC-AUC"] = np.nan
        metrics["PR-AUC"] = np.nan

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Searches for optimal decision threshold on validation probabilities to maximize specified metric.
    Default standard threshold is 0.50, but for rare classes (imbalance), optimal threshold is often 0.20-0.35.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.50
    best_score = -1.0

    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, preds, zero_division=0)
        elif metric == "macro_f1":
            score = f1_score(y_true, preds, average="macro", zero_division=0)
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return round(float(best_threshold), 3), round(float(best_score), 4)


def plot_confusion_matrices(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Path] = None,
):
    """
    Plots confusion matrices side-by-side for top candidate models.
    results_dict format: {model_name: (y_true, y_pred)}
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)

    for idx, (model_name, (y_true, y_pred)) in enumerate(results_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Serious", "Serious"])
        disp.plot(ax=axes[0, idx], cmap="Blues", colorbar=False)
        axes[0, idx].set_title(f"{model_name}\nConfusion Matrix", fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Evaluation] Saved confusion matrices to {save_path}")
    plt.close(fig)


def plot_roc_and_pr_curves(
    curves_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Path] = None,
):
    """
    Plots ROC and Precision-Recall curves comparing multiple models.
    curves_dict format: {model_name: (y_true, y_prob)}
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    for model_name, (y_true, y_prob) in curves_dict.items():
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

        # PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax_pr.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")

    # Baseline references
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Baseline")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    # PR curve baseline (no-skill rate = prevalence of positive class)
    first_y = next(iter(curves_dict.values()))[0]
    baseline_pr = np.mean(first_y)
    ax_pr.plot([0, 1], [baseline_pr, baseline_pr], "k--", alpha=0.6, label=f"Baseline ({baseline_pr:.2f})")
    ax_pr.set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall (Serious Accidents)", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.legend(loc="best")
    ax_pr.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Evaluation] Saved ROC & PR curves to {save_path}")
    plt.close(fig)
