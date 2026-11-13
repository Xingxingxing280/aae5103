from __future__ import annotations

import html
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import predictions_dir, reports_dir
from src.predict import artifacts_exist, load_artifacts, predict_schedule

DAY_NAMES = {
    1: "周一",
    2: "周二",
    3: "周三",
    4: "周四",
    5: "周五",
    6: "周六",
    7: "周日",
}


def _page(body: str, title: str = "航班出发延误风险演示") -> bytes:
    document = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 24px; background: #f5f7fb; color: #1d2433; }}
    h1, h2 {{ margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: 360px 1fr; gap: 20px; }}
    .card {{ background: white; padding: 18px 20px; border-radius: 12px; box-shadow: 0 1px 4px rgba(16,24,40,.08); }}
    label {{ display: block; font-size: 13px; margin: 10px 0 4px; }}
    input, select, button {{ width: 100%; padding: 8px 10px; box-sizing: border-box; }}
    button {{ background: #1f4e79; color: white; border: 0; border-radius: 8px; margin-top: 14px; cursor: pointer; }}
    .metric {{ font-size: 28px; font-weight: 700; }}
    .Low {{ color: #1b7f4e; }}
    .Medium {{ color: #b7791f; }}
    .High {{ color: #b42318; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5eaf2; padding: 6px 8px; text-align: left; }}
    .muted {{ color: #667085; font-size: 13px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
    return document.encode("utf-8")


def _metrics_html() -> str:
    valid = reports_dir() / "metrics_validation.csv"
    test = reports_dir() / "metrics_test.csv"
    if not valid.exists() or not test.exists():
        return "<p>尚未生成评价结果，请先运行 <code>python scripts/train_models.py</code>。</p>"
    chunks = ["<h2>验证集</h2>", _csv_table(valid), "<h2>测试集</h2>", _csv_table(test)]
    bands = reports_dir() / "risk_bands_test.csv"
    if bands.exists():
        chunks.extend(["<h2>测试集风险等级实际延误率</h2>", _csv_table(bands)])
    return "\n".join(chunks)


def _csv_table(path: Path) -> str:
    import pandas as pd

    frame = pd.read_csv(path)
    keep = [
        column
        for column in frame.columns
        if column
        in {
            "split",
            "model",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "risk_level",
            "n",
            "share",
            "observed_delay_rate",
            "mean_predicted_probability",
        }
    ]
    view = frame[keep] if keep else frame

    def _fmt(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in view.columns)
    rows = []
    for record in view.head(12).to_dict(orient="records"):
        cells = "".join(f"<td>{html.escape(_fmt(value))}</td>" for value in record.values())
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><tr>{header}</tr>{''.join(rows)}</table>"


def _sample_table() -> str:
    path = predictions_dir() / "test.csv"
    if not path.exists():
        return ""
    import pandas as pd

    sample = pd.read_csv(path).sample(12, random_state=42)
    columns = [
        "FlightDate",
        "Airline",
        "FlightNumber",
        "Origin",
        "Dest",
        "CRSDepTime",
        "delay_probability",
        "risk_level",
        "TargetDelayed15",
    ]
    view = sample[columns]
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    rows = []
    for record in view.to_dict(orient="records"):
        record["delay_probability"] = f"{record['delay_probability']:.3f}"
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in record.values())
        rows.append(f"<tr>{cells}</tr>")
    return "<h2>测试集抽样预测</h2><table><tr>" + header + "</tr>" + "".join(rows) + "</table>"


def _result_card(result: dict | None, error: str | None) -> str:
    if error:
        return f'<div class="card"><h2>预测结果</h2><p class="High">{html.escape(error)}</p></div>'
    if result is None:
        return '<div class="card"><h2>预测结果</h2><p class="muted">在左侧输入计划航班信息后提交。不要填写实际出发时间或延误分钟数。</p></div>'
    risk = result["risk_level"]
    predicted = "延误 ≥ 15 分钟" if result["predicted_delayed15"] else "预计不足 15 分钟"
    return f"""
    <div class="card">
      <h2>预测结果</h2>
      <p class="muted">模型：{html.escape(str(result['model']))}；决策阈值 {result['decision_threshold']:.3f}</p>
      <p>延误概率</p>
      <p class="metric">{result['delay_probability']:.1%}</p>
      <p>风险等级：<span class="{html.escape(risk)}">{html.escape(risk)}</span></p>
      <p>分类结果：{predicted}</p>
      <p class="muted">计划特征：{html.escape(result['Airline'])} {html.escape(result['Origin'])}→{html.escape(result['Dest'])}，星期 {result['DayOfWeek']}，当地分钟 {result['ScheduledMinuteOfDay']}。</p>
    </div>
    """


def _form(defaults: dict) -> str:
    return f"""
    <form method="get" action="/">
      <label>航空公司代码 Airline</label>
      <input name="airline" value="{html.escape(defaults['airline'])}" required>
      <label>出发机场 Origin</label>
      <input name="origin" value="{html.escape(defaults['origin'])}" required>
      <label>到达机场 Dest</label>
      <input name="dest" value="{html.escape(defaults['dest'])}" required>
      <label>星期 DayOfWeek（1=周一，7=周日）</label>
      <select name="day_of_week">
        {''.join(f'<option value="{day}" {"selected" if str(day)==str(defaults["day_of_week"]) else ""}>{day} {DAY_NAMES[day]}</option>' for day in range(1,8))}
      </select>
      <label>计划出发时间 CRSDepTime（HHMM）</label>
      <input name="crs_dep_time" value="{html.escape(str(defaults['crs_dep_time']))}" required>
      <label>距离 Distance（英里）</label>
      <input name="distance" value="{html.escape(str(defaults['distance']))}" required>
      <label>计划航程 CRSElapsedTime（分钟）</label>
      <input name="crs_elapsed_time" value="{html.escape(str(defaults['crs_elapsed_time']))}" required>
      <button type="submit" name="predict" value="1">预测延误风险</button>
    </form>
    """


def _home(query: dict[str, list[str]]) -> bytes:
    defaults = {
        "airline": "AA",
        "origin": "DFW",
        "dest": "PHX",
        "day_of_week": "3",
        "crs_dep_time": "1400",
        "distance": "868",
        "crs_elapsed_time": "160",
    }
    for key in defaults:
        if key in query and query[key][0]:
            defaults[key] = query[key][0]

    result = None
    error = None
    submitted = "airline" in query
    if submitted:
        try:
            artifacts = load_artifacts()
            result = predict_schedule(
                artifacts,
                airline=defaults["airline"].strip().upper(),
                origin=defaults["origin"].strip().upper(),
                dest=defaults["dest"].strip().upper(),
                day_of_week=int(defaults["day_of_week"]),
                crs_dep_time=int(defaults["crs_dep_time"]),
                distance=float(defaults["distance"]),
                crs_elapsed_time=float(defaults["crs_elapsed_time"]),
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

    body = f"""
    <h1>出发延误风险演示</h1>
    <p class="muted">仅使用计划信息预测登机口出发是否延误至少 15 分钟。数据范围：2025 年 1–3 月，出发机场 ATL / ORD / DFW。</p>
    <div class="grid">
      <div class="card">
        <h2>计划航班输入</h2>
        {_form(defaults)}
      </div>
      {_result_card(result, error)}
    </div>
    <div class="card" style="margin-top:20px">{_metrics_html()}{_sample_table()}</div>
    """
    return _page(body)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/", "/index.html"}:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return
        payload = _home(parse_qs(parsed.query))
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    if not artifacts_exist():
        raise SystemExit("请先运行 python scripts/train_models.py")
    host, port = "127.0.0.1", 8765
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"打开浏览器访问 http://{host}:{port}", flush=True)
    print("按 Ctrl+C 结束", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
