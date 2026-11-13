from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    split: str,
    model: str,
    threshold: float,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        "split": split,
        "model": model,
        "threshold": threshold,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "delay_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if y_proba is not None:
        row["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        row["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan
    return row


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def majority_baseline(y_true: np.ndarray, *, split: str) -> dict:
    y_pred = np.zeros_like(y_true)
    return classification_report_row(
        y_true,
        y_pred,
        None,
        split=split,
        model="majority_on_time",
        threshold=1.0,
    )


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "split",
        "model",
        "threshold",
        "n",
        "positives",
        "delay_rate",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    return pd.DataFrame(rows)[columns]
