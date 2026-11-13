from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import data_root, load_config


def _read_split(split: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    processed = data_root()
    x = pd.read_csv(processed / f"X_{split}.csv")
    y = pd.read_csv(processed / f"y_{split}.csv")["TargetDelayed15"].astype(int)
    metadata = pd.read_csv(processed / f"metadata_{split}.csv")
    if not (len(x) == len(y) == len(metadata)):
        raise ValueError(f"{split} 的 X/y/metadata 行数不一致")
    config = load_config()
    expected = config["categorical_features"] + config["numeric_features"]
    missing = [column for column in expected if column not in x.columns]
    extra = [column for column in x.columns if column not in expected]
    if missing or extra:
        raise ValueError(f"{split} 特征列不符合白名单：缺少 {missing}，多余 {extra}")
    return x, y, metadata


def load_splits() -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    return {split: _read_split(split) for split in ("train", "validation", "test")}


def processed_exists() -> bool:
    processed = data_root()
    needed = [
        "X_train.csv",
        "y_train.csv",
        "X_validation.csv",
        "y_validation.csv",
        "X_test.csv",
        "y_test.csv",
        "metadata_test.csv",
    ]
    return all((processed / name).exists() for name in needed)


def find_handoff_root() -> Path:
    return data_root().parents[1]
