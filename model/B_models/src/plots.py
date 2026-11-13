from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from .paths import figures_dir


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["On-time", "Delayed ≥15m"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figures_dir() / filename, dpi=140)
    plt.close(fig)


def save_roc_pr(y_true: np.ndarray, scores: dict[str, np.ndarray], filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for name, proba in scores.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        precision, recall, _ = precision_recall_curve(y_true, proba)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir() / filename, dpi=140)
    plt.close(fig)


def save_metrics_bars(table: pd.DataFrame, filename: str) -> None:
    plot_df = table[table["model"] != "majority_on_time"].copy()
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(plot_df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, plot_df[metric], width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Validation Precision / Recall / F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir() / filename, dpi=140)
    plt.close(fig)


def save_risk_bars(summary: pd.DataFrame, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(summary["risk_level"], summary["observed_delay_rate"])
    ax.set_ylabel("Observed delay rate")
    ax.set_title(title)
    ax.set_ylim(0, max(0.5, float(summary["observed_delay_rate"].max()) + 0.05))
    fig.tight_layout()
    fig.savefig(figures_dir() / filename, dpi=140)
    plt.close(fig)
