"""Build documentation from actual run outputs; never fill in fabricated statistics."""
import csv
import json
from pathlib import Path
import platform
import zipfile
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build():
    docs = ROOT / "docs"
    sources = docs / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    report = json.loads((ROOT / "reports/manifest.json").read_text())
    checks = json.loads((ROOT / "reports/verification.json").read_text())
    if not checks["successful"]:
        raise ValueError("Cannot publish handoff with failed checks")
    for meta in report["sources"]:
        name = meta["url"].rsplit("/", 1)[-1]
        with zipfile.ZipFile(ROOT / "data/raw" / name) as z:
            (sources / (name.removesuffix(".zip") + "_readme.html")).write_bytes(z.read("readme.html"))
    specs = {
        "record_id": ("项目生成", "航班标识字段拼接的 SHA-256 前24位；唯一性已检查", "连接预测结果，不入模型"),
        "Year": ("Year", "航班年份", "审计"),
        "Month": ("Month", "月份1–12", "描述统计"),
        "DayofMonth": ("DayofMonth", "日期中的日", "审计"),
        "DayOfWeek": ("DayOfWeek", "周一1至周日7，与日期核对", "模型特征"),
        "FlightDate": ("FlightDate", "当地航班日期；YYYY-MM-DD", "划分、描述统计"),
        "Airline": ("Reporting_Airline", "报告承运人代码，不是航空公司全名或营销承运人", "模型类别特征"),
        "FlightNumber": ("Flight_Number_Reporting_Airline", "航班号，按字符串保存", "身份核对、不入模型"),
        "Origin": ("Origin", "出发机场代码", "模型类别特征"),
        "Dest": ("Dest", "目的机场代码", "模型类别特征"),
        "CRSDepTime": ("CRSDepTime", "当地计划登机口出发时间，HHMM，非分钟数", "生成计划时段特征"),
        "DepTime": ("DepTime", "当地实际登机口出发时间，HHMM，可缺失", "事后信息，禁止入模型"),
        "DepDelay": ("DepDelay", "出发延误分钟数，负值为提前出发", "标签核对、禁止入模型"),
        "DepDel15": ("DepDel15", "官方出发延误至少15分钟指示；1=是", "答案字段，禁止入模型"),
        "Cancelled": ("Cancelled", "取消指示；1=取消", "描述统计与样本筛选，禁止入模型"),
        "Diverted": ("Diverted", "备降指示；1=是", "审计，禁止入模型"),
        "Distance": ("Distance", "机场间距离，英里", "模型数值特征"),
        "CRSElapsedTime": ("CRSElapsedTime", "计划航程时间，分钟", "模型数值特征"),
        "SourceArchive": ("项目生成", "来源ZIP文件名", "追溯"),
        "SourceMember": ("项目生成", "ZIP内部CSV文件名", "追溯"),
        "SourceRow": ("项目生成", "原始CSV从1开始的数据行序号，不包括表头", "追溯"),
        "TargetDelayed15": ("由DepDel15筛选", "非取消航班标签0/1；取消或无标签记录为空", "目标y，禁止放入X"),
        "ScheduledMinuteOfDay": ("由CRSDepTime生成", "计划当地分钟数0–1439；2400映射0", "模型数值特征"),
        "ScheduledHour": ("由CRSDepTime生成", "计划当地小时0–23", "模型数值特征"),
        "IsWeekend": ("由DayOfWeek生成", "周六/周日为1，否则0", "模型数值特征"),
        "ScheduledAt2400": ("由CRSDepTime生成", "原始计划时间是否等于2400", "审计"),
        "ActualTimeInvalid": ("由DepTime生成", "非空实际时间不符合HHMM规则为1；空值另见missingness", "审计，禁止入模型"),
        "split": ("由FlightDate生成", "train / validation / test；模型排除记录也有日期所属划分", "分组信息，禁止入模型"),
        "ModelExclusionReason": ("项目规则", "首个模型排除原因；空字符串表示无排除原因", "审计"),
        "ModelEligible": ("项目规则", "是否满足当前建模样本规则", "筛选，禁止入模型"),
    }
    d = pd.read_csv(ROOT / "data/processed/cleaned_flights.csv")
    if set(specs) != set(d.columns):
        raise ValueError("Dictionary does not match actual output schema")
    with (docs / "数据字典.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["字段", "来源", "定义与单位", "用途", "本次读入类型", "缺失数量"])
        for field in d:
            w.writerow([field, *specs[field], str(d[field].dtype), int(d[field].isna().sum())])
    (ROOT / "requirements-run.txt").write_text(
        f"# Actual successful run: Python {platform.python_version()}\npandas=={pd.__version__}\nnumpy=={np.__version__}\n", encoding="utf-8")
    cancelled = int(d.Cancelled.sum())
    target = d.TargetDelayed15.dropna()
    text = f"""# 本次数据质量与交付结果

下列数值由本次实际输出计算；运行时间为 {report['generated_at_utc']}（UTC）。

## 数据规模

- 原始三个月CSV合计：{sum(s['total_rows'] for s in report['sources']):,} 行。
- 三个出发机场筛选后：{len(d):,} 行（分析版，包含取消）。
- 取消航班：{cancelled:,} 行；取消占比：{cancelled/len(d):.4%}。
- 可建模航班：{report['rows_model']:,} 行。
- 有效标签中延误至少15分钟：{int(target.sum()):,} 行；延误率 {target.mean():.4%}。
- 非取消、备降航班：{int((d.Diverted.eq(1)&d.Cancelled.eq(0)).sum()):,} 行，出发标签有效者保留。
- 计划时间为2400：{int(d.ScheduledAt2400.sum()):,} 行。
- 非空实际时间格式异常：{int(d.ActualTimeInvalid.sum()):,} 行。

## 固定时间划分

| 集合 | 日期范围 | 样本量 | 延误样本 | 延误率 |
|---|---|---:|---:|---:|
"""
    for s in report["splits"]:
        text += f"| {s['split']} | {s['start']} 至 {s['end']} | {s['rows']:,} | {s['delayed']:,} | {s['delay_rate']:.4%} |\n"
    text += f"""
## 核对结果

已执行 {checks['tests_run']} 项测试，失败 {checks['failures']}，错误 {checks['errors']}。检查时间 {checks['checked_at_utc']}（UTC）。

- 原始ZIP下载长度、ZIP成员CRC及后续SHA-256核对。
- 原始月度日期范围、日期与星期等日历字段一致性。
- 所有非取消且有标签记录的 DepDel15 与 DepDelay >= 15 一致。
- **逐字段回查所有保留记录**与ZIP原始CSV，包括数值及缺失值，而非只检查几条样例。
- 航班身份冲突、record_id唯一性、时间切分、集合之间ID不重复。
- X列严格符合白名单；X、y、metadata逐行对应；全部符合模型条件的记录恰好覆盖一次。
- HHMM、2400、非法分钟、负时间、15分钟边界、取消标签、错误标签和错误星期等构造边界案例。这些构造数据只用于测试，未混入研究数据。
- 输出CSV哈希、清洗审计行数守恒。

具体结果见 verification.json、manifest.json、cleaning_audit.csv、missingness.csv。第一轮独立回查发现比较程序对 NaN 与 pandas.NA 的表示不一致；已统一核对程序的缺失值类型后重新通过，未据此改动研究数据。

取消记录的空标签和空实际出发值是有意义的缺失，不填造0。ModelExclusionReason 对可建模记录为空字符串，CSV读入后可能显示为NaN，这不是航班属性缺失。

这里没有训练或评价AI模型，也没有测得任何运营干预收益；延误率是数据描述，不是预测成绩。检查通过不证明BTS原始报送数据绝对无误。机场、时期和时间划分为本版明确记录的研究选择，详见方法说明。
"""
    (ROOT / "reports/数据质量报告.md").write_text(text, encoding="utf-8")
    cells = []
    def md(s):
        cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
    def code(s):
        cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})
    md("# 成员 A 数据交接演示\n本 Notebook 用于查看真实交付数据，不包含训练成绩。支持从根目录或 notebooks 目录启动。")
    code("from pathlib import Path\nimport pandas as pd\nroot = Path.cwd()\nif not (root / 'config.json').exists():\n    root = root.parent\nassert (root / 'config.json').exists(), '请从项目或notebooks目录启动'\np = root / 'data' / 'processed'\nd = pd.read_csv(p / 'cleaned_flights.csv')\nd.shape, d.head()")
    md("## 完整时期的描述统计\n这些统计用于报告说明，不允许回填为预测特征。取消航班不计入延误率分母。")
    code("pd.read_csv(root / 'reports' / 'summary_airport.csv')")
    code("pd.read_csv(root / 'reports' / 'split_summary.csv')")
    md("## B 读取训练数据\nX中只有计划相关信息；标签单独读取。训练用编码和标准化由B在Pipeline中拟合。")
    code("X_train = pd.read_csv(p / 'X_train.csv')\ny_train = pd.read_csv(p / 'y_train.csv')['TargetDelayed15']\nmetadata_train = pd.read_csv(p / 'metadata_train.csv')\nassert len(X_train) == len(y_train) == len(metadata_train)\nX_train.head()")
    md("## 质量检查记录与来源\n详见 docs/方法与局限.md、docs/交接给B.md 及 reports/manifest.json。")
    code("import json\njson.loads((root / 'reports' / 'verification.json').read_text())")
    notebook = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    folder = ROOT / "notebooks"
    folder.mkdir(exist_ok=True)
    for i, cell in enumerate(cells):
        cell["id"] = f"handoff-{i:02}"
    (folder / "01_data_handoff.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Built field dictionary, official readmes, quality report, environment snapshot and handoff notebook.")


if __name__ == "__main__":
    build()
