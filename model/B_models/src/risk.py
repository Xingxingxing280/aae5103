from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import load_config, models_dir


def assign_risk(probability: np.ndarray, medium_cut: float, high_cut: float) -> np.ndarray:
    labels = np.full(len(probability), "Low", dtype=object)
    labels[probability >= medium_cut] = "Medium"
    labels[probability >= high_cut] = "High"
    return labels


def fit_risk_policy(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    config = load_config()
    medium_q = config["risk_quantiles"]["medium"]
    high_q = config["risk_quantiles"]["high"]
    medium_cut = float(np.quantile(y_proba, medium_q))
    high_cut = float(np.quantile(y_proba, high_q))
    if high_cut <= medium_cut:
        high_cut = min(0.999, medium_cut + 1e-4)
    labels = assign_risk(y_proba, medium_cut, high_cut)
    band_rows = []
    for band in ("Low", "Medium", "High"):
        mask = labels == band
        band_rows.append(
            {
                "risk_level": band,
                "n": int(mask.sum()),
                "share": float(mask.mean()),
                "observed_delay_rate": float(y_true[mask].mean()) if mask.any() else np.nan,
                "mean_predicted_probability": float(y_proba[mask].mean())
                if mask.any()
                else np.nan,
            }
        )
    return {
        "medium_cut": medium_cut,
        "high_cut": high_cut,
        "quantiles": {"medium": medium_q, "high": high_q},
        "validation_bands": band_rows,
        "note": (
            "风险切分来自验证集预测概率分位数，冻结后再用于测试集。"
            "High 对应验证集概率最高的 20%，Medium 为中间 40%，Low 为最低 40%。"
            "风险等级不是因果干预效果，也不等于官方延误标签。"
        ),
    }


def save_risk_policy(policy: dict) -> Path:
    path = models_dir() / "risk_policy.json"
    path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_risk_policy() -> dict:
    path = models_dir() / "risk_policy.json"
    return json.loads(path.read_text(encoding="utf-8"))


def band_summary(y_true: np.ndarray, y_proba: np.ndarray, policy: dict) -> pd.DataFrame:
    labels = assign_risk(y_proba, policy["medium_cut"], policy["high_cut"])
    rows = []
    for band in ("Low", "Medium", "High"):
        mask = labels == band
        rows.append(
            {
                "risk_level": band,
                "n": int(mask.sum()),
                "share": float(mask.mean()),
                "observed_delay_rate": float(y_true[mask].mean()) if mask.any() else np.nan,
                "mean_predicted_probability": float(y_proba[mask].mean())
                if mask.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
