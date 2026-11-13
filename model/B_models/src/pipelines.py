from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .paths import load_config


def _categorical_encoder(*, dense: bool) -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)


def make_logistic_regression(*, class_weight: str | None, random_state: int) -> Pipeline:
    config = load_config()
    preprocess = ColumnTransformer(
        [
            (
                "cat",
                _categorical_encoder(dense=False),
                config["categorical_features"],
            ),
            ("num", StandardScaler(), config["numeric_features"]),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_random_forest(*, class_weight: str | None, random_state: int) -> Pipeline:
    config = load_config()
    preprocess = ColumnTransformer(
        [
            (
                "cat",
                _categorical_encoder(dense=True),
                config["categorical_features"],
            ),
            ("num", "passthrough", config["numeric_features"]),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=14,
                    min_samples_leaf=4,
                    n_jobs=-1,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def candidate_specs(random_state: int) -> list[dict]:
    """验证集上比较的候选模型。超参数在此固定，不根据测试集挑选。"""
    return [
        {
            "name": "logistic_regression",
            "display": "Logistic Regression",
            "family": "logistic_regression",
            "class_weight": None,
            "builder": lambda: make_logistic_regression(
                class_weight=None, random_state=random_state
            ),
        },
        {
            "name": "logistic_regression_balanced",
            "display": "Logistic Regression (balanced)",
            "family": "logistic_regression",
            "class_weight": "balanced",
            "builder": lambda: make_logistic_regression(
                class_weight="balanced", random_state=random_state
            ),
        },
        {
            "name": "random_forest",
            "display": "Random Forest",
            "family": "random_forest",
            "class_weight": None,
            "builder": lambda: make_random_forest(
                class_weight=None, random_state=random_state
            ),
        },
        {
            "name": "random_forest_balanced",
            "display": "Random Forest (balanced)",
            "family": "random_forest",
            "class_weight": "balanced",
            "builder": lambda: make_random_forest(
                class_weight="balanced", random_state=random_state
            ),
        },
    ]
