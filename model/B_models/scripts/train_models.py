from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import pandas as pd
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_splits
from src.metrics import (
    best_f1_threshold,
    classification_report_row,
    majority_baseline,
    metrics_table,
)
from src.paths import figures_dir, load_config, models_dir, predictions_dir, reports_dir
from src.pipelines import candidate_specs
from src.plots import save_confusion, save_metrics_bars, save_risk_bars, save_roc_pr
from src.predict import attach_predictions, load_artifacts, score_frame
from src.risk import band_summary, fit_risk_policy, save_risk_policy


REQUIRED_FAMILIES = ("logistic_regression", "random_forest")


def _fit_candidates(x_train, y_train, x_valid, y_valid, random_state: int):
    fitted = {}
    valid_rows = []
    for spec in candidate_specs(random_state):
        print(f"训练 {spec['display']} ...")
        model = spec["builder"]()
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_valid)[:, 1]
        threshold, _ = best_f1_threshold(y_valid.to_numpy(), proba)
        pred = (proba >= threshold).astype(int)
        row = classification_report_row(
            y_valid.to_numpy(),
            pred,
            proba,
            split="validation",
            model=spec["name"],
            threshold=threshold,
        )
        row["display"] = spec["display"]
        row["family"] = spec["family"]
        valid_rows.append(row)
        fitted[spec["name"]] = {
            "spec": spec,
            "model": model,
            "threshold": threshold,
            "proba": proba,
            "pred": pred,
        }
        print(
            f"  F1={row['f1']:.4f}  Precision={row['precision']:.4f}  "
            f"Recall={row['recall']:.4f}  threshold={threshold:.4f}"
        )
    return fitted, valid_rows


def _pick_selected(valid_rows: list[dict]) -> dict:
    ranked = sorted(valid_rows, key=lambda row: (row["f1"], row["pr_auc"]), reverse=True)
    return ranked[0]


def _pick_family_best(valid_rows: list[dict], family: str) -> dict:
    family_rows = [row for row in valid_rows if row["family"] == family]
    return sorted(family_rows, key=lambda row: (row["f1"], row["pr_auc"]), reverse=True)[0]


def _markdown_table(table: pd.DataFrame) -> str:
    rounded = table.copy()
    for column in rounded.columns:
        if pd.api.types.is_float_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda value: f"{value:.4f}")
    headers = list(rounded.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rounded.itertuples(index=False):
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _write_markdown(valid_table: pd.DataFrame, test_table: pd.DataFrame, selection: dict) -> None:
    path = reports_dir() / "model_comparison.md"
    lines = [
        "# 模型对比结果",
        "",
        "决策阈值在验证集上按正类 F1 最大选取，测试集只用于最终报告。",
        "不要把 Accuracy 当作主要指标：多数航班并不延误，全预测不延误也会得到较高准确率。",
        "",
        f"- 选定模型：`{selection['selected_model']}`",
        f"- 验证集 F1：{selection['validation_f1']:.4f}",
        f"- 决策阈值：{selection['decision_threshold']:.4f}",
        "",
        "## 验证集",
        "",
        _markdown_table(valid_table),
        "",
        "## 测试集（最终评价）",
        "",
        _markdown_table(test_table),
        "",
        "## 使用说明",
        "",
        "- 预测界面只能输入计划字段，不能输入实际出发时间或延误分钟数。",
        "- 风险等级来自验证集概率分位数，不是官方延误标签。",
        "- 分类指标不能直接解释为减少了多少延误。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_config()
    random_state = int(config["random_state"])
    splits = load_splits()
    x_train, y_train, _ = splits["train"]
    x_valid, y_valid, meta_valid = splits["validation"]
    x_test, y_test, meta_test = splits["test"]

    fitted, valid_rows = _fit_candidates(x_train, y_train, x_valid, y_valid, random_state)
    selected_row = _pick_selected(valid_rows)
    selected_name = selected_row["model"]
    selected_pack = fitted[selected_name]
    lr_row = _pick_family_best(valid_rows, "logistic_regression")
    rf_row = _pick_family_best(valid_rows, "random_forest")
    valid_rows.append(majority_baseline(y_valid.to_numpy(), split="validation"))
    valid_table = metrics_table(valid_rows)
    valid_table.to_csv(reports_dir() / "metrics_validation.csv", index=False)

    for name, pack in fitted.items():
        joblib.dump(pack["model"], models_dir() / f"{name}.joblib")
    joblib.dump(selected_pack["model"], models_dir() / "selected_model.joblib")

    policy = fit_risk_policy(y_valid.to_numpy(), selected_pack["proba"])
    save_risk_policy(policy)

    selection = {
        "selected_model": selected_name,
        "selected_display": selected_row["display"],
        "decision_threshold": float(selected_row["threshold"]),
        "validation_f1": float(selected_row["f1"]),
        "validation_precision": float(selected_row["precision"]),
        "validation_recall": float(selected_row["recall"]),
        "family_best": {
            "logistic_regression": lr_row["model"],
            "random_forest": rf_row["model"],
        },
        "family_thresholds": {
            "logistic_regression": float(lr_row["threshold"]),
            "random_forest": float(rf_row["threshold"]),
        },
        "positive_class": "TargetDelayed15 = 1, gate departure delay of at least 15 minutes",
        "features": config["categorical_features"] + config["numeric_features"],
    }
    (models_dir() / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    artifacts = load_artifacts()
    test_rows = [majority_baseline(y_test.to_numpy(), split="test")]
    test_probas = {}
    for name, pack in fitted.items():
        proba = pack["model"].predict_proba(x_test)[:, 1]
        pred = (proba >= pack["threshold"]).astype(int)
        test_rows.append(
            classification_report_row(
                y_test.to_numpy(),
                pred,
                proba,
                split="test",
                model=name,
                threshold=pack["threshold"],
            )
        )
        if name in REQUIRED_FAMILIES:
            test_probas[pack["spec"]["display"]] = proba
            save_confusion(
                y_test.to_numpy(),
                pred,
                f"Test confusion: {pack['spec']['display']}",
                f"confusion_test_{name}.png",
            )
    selected_test = score_frame(
        artifacts["models"]["selected"],
        x_test,
        threshold=artifacts["threshold"],
        policy=artifacts["policy"],
    )
    test_table = metrics_table(test_rows)
    test_table.to_csv(reports_dir() / "metrics_test.csv", index=False)

    pred_valid = attach_predictions(meta_valid, x_valid, y_valid, artifacts)
    pred_test = attach_predictions(meta_test, x_test, y_test, artifacts)
    pred_valid.to_csv(predictions_dir() / "validation.csv", index=False)
    pred_test.to_csv(predictions_dir() / "test.csv", index=False)

    valid_bands = band_summary(y_valid.to_numpy(), selected_pack["proba"], policy)
    test_bands = band_summary(
        y_test.to_numpy(), selected_test["delay_probability"].to_numpy(), policy
    )
    valid_bands.assign(split="validation").to_csv(
        reports_dir() / "risk_bands_validation.csv", index=False
    )
    test_bands.assign(split="test").to_csv(reports_dir() / "risk_bands_test.csv", index=False)

    comparison_valid = valid_table[valid_table["model"] != "majority_on_time"].copy()
    save_metrics_bars(comparison_valid, "validation_prf.png")
    save_roc_pr(
        y_valid.to_numpy(),
        {
            fitted[name]["spec"]["display"]: fitted[name]["proba"]
            for name in REQUIRED_FAMILIES
        },
        "validation_roc_pr.png",
    )
    save_roc_pr(y_test.to_numpy(), test_probas, "test_roc_pr.png")
    save_risk_bars(valid_bands, "Validation delay rate by risk band", "risk_validation.png")
    save_risk_bars(test_bands, "Test delay rate by risk band", "risk_test.png")
    save_confusion(
        y_valid.to_numpy(),
        selected_pack["pred"],
        f"Validation confusion: {selected_name}",
        "confusion_validation_selected.png",
    )

    _write_markdown(valid_table, test_table, selection)
    report_lines = [
        "sklearn classification_report",
        "正类 = TargetDelayed15（登机口出发延误至少 15 分钟）",
        "阈值在验证集上按 F1 选定，测试集不再调阈值。",
        "",
    ]
    for split_name, y_true, source in (
        ("validation", y_valid.to_numpy(), fitted),
        ("test", y_test.to_numpy(), fitted),
    ):
        for family in REQUIRED_FAMILIES:
            pack = source[family]
            if split_name == "validation":
                pred = pack["pred"]
            else:
                proba = pack["model"].predict_proba(x_test)[:, 1]
                pred = (proba >= pack["threshold"]).astype(int)
            report_lines.append(f"## {split_name} / {pack['spec']['display']}")
            report_lines.append(
                classification_report(
                    y_true,
                    pred,
                    labels=[0, 1],
                    target_names=["on_time", "delayed_15"],
                    digits=4,
                    zero_division=0,
                )
            )
    (reports_dir() / "classification_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print("选定模型:", selected_name)
    print("验证集 F1:", f"{selected_row['f1']:.4f}")
    print("结果写入 reports/、models/、data/predictions/")
    print("图片:", sorted(path.name for path in figures_dir().glob("*.png")))


if __name__ == "__main__":
    main()
