from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import build_feature_row
from .paths import models_dir
from .risk import assign_risk, load_risk_policy


def load_artifacts() -> dict:
    directory = models_dir()
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    models = {
        "logistic_regression": joblib.load(directory / "logistic_regression.joblib"),
        "random_forest": joblib.load(directory / "random_forest.joblib"),
        "selected": joblib.load(directory / "selected_model.joblib"),
    }
    return {
        "selection": selection,
        "models": models,
        "policy": load_risk_policy(),
        "threshold": float(selection["decision_threshold"]),
        "selected_name": selection["selected_model"],
    }


def score_frame(model, features: pd.DataFrame, *, threshold: float, policy: dict) -> pd.DataFrame:
    probability = model.predict_proba(features)[:, 1]
    predicted = (probability >= threshold).astype(int)
    risk = assign_risk(probability, policy["medium_cut"], policy["high_cut"])
    return pd.DataFrame(
        {
            "delay_probability": probability,
            "predicted_delayed15": predicted,
            "risk_level": risk,
        }
    )


def predict_schedule(
    artifacts: dict,
    *,
    airline: str,
    origin: str,
    dest: str,
    day_of_week: int,
    crs_dep_time: int,
    distance: float,
    crs_elapsed_time: float,
    model_name: str = "selected",
) -> dict:
    features = build_feature_row(
        airline=airline,
        origin=origin,
        dest=dest,
        day_of_week=day_of_week,
        crs_dep_time=crs_dep_time,
        distance=distance,
        crs_elapsed_time=crs_elapsed_time,
    )
    model = artifacts["models"][model_name]
    scored = score_frame(
        model,
        features,
        threshold=artifacts["threshold"],
        policy=artifacts["policy"],
    ).iloc[0]
    return {
        **features.iloc[0].to_dict(),
        "model": artifacts["selected_name"] if model_name == "selected" else model_name,
        "decision_threshold": artifacts["threshold"],
        "delay_probability": float(scored["delay_probability"]),
        "predicted_delayed15": int(scored["predicted_delayed15"]),
        "risk_level": str(scored["risk_level"]),
    }


def attach_predictions(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    y_true: pd.Series | None,
    artifacts: dict,
    *,
    model_name: str = "selected",
) -> pd.DataFrame:
    scored = score_frame(
        artifacts["models"][model_name],
        features,
        threshold=artifacts["threshold"],
        policy=artifacts["policy"],
    )
    out = metadata.copy()
    out["delay_probability"] = scored["delay_probability"].to_numpy()
    out["predicted_delayed15"] = scored["predicted_delayed15"].to_numpy()
    out["risk_level"] = scored["risk_level"].to_numpy()
    out["decision_threshold"] = artifacts["threshold"]
    out["model"] = artifacts["selected_name"] if model_name == "selected" else model_name
    if y_true is not None:
        out["TargetDelayed15"] = y_true.to_numpy()
    return out


def artifacts_exist() -> bool:
    directory = models_dir()
    needed = [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "selected_model.joblib",
        "selection.json",
        "risk_policy.json",
    ]
    return all((directory / name).exists() for name in needed)


def selection_path() -> Path:
    return models_dir() / "selection.json"
