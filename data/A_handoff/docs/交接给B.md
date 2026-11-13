# 给成员 B 的交接说明

## 输入文件

每个 split（train / validation / test）包含三份按行对齐的 CSV：

- `X_{split}.csv`：只包含允许使用的 9 个计划信息特征。
- `y_{split}.csv`：一列 `TargetDelayed15`，1 表示出发延误至少 15 分钟，0 表示不足 15 分钟，包括提前出发。
- `metadata_{split}.csv`：record_id、日期、航空公司、航班号、机场、计划时间，用于回接预测结果，不整体送进模型。

例子（以项目根目录为当前目录）：

```python
import pandas as pd
from pathlib import Path
p = Path('data/processed')
X_train = pd.read_csv(p / 'X_train.csv')
y_train = pd.read_csv(p / 'y_train.csv')['TargetDelayed15']
X_validation = pd.read_csv(p / 'X_validation.csv')
y_validation = pd.read_csv(p / 'y_validation.csv')['TargetDelayed15']
X_test = pd.read_csv(p / 'X_test.csv')
y_test = pd.read_csv(p / 'y_test.csv')['TargetDelayed15']
metadata_test = pd.read_csv(p / 'metadata_test.csv')
# 完成预测后：
# predictions = metadata_test.assign(delay_probability=model.predict_proba(X_test)[:, 1])
```

`model_data.csv` 是相同数据的合并版，额外带 record_id、FlightDate、split、标签；不要把这些列自动当作输入。`cleaned_flights.csv` 是分析版，包含答案和实际运行信息，不能直接用“删除标签后的剩余所有列”建模。

## 特征处理

类别列：Airline、Origin、Dest。由 B 在训练 Pipeline 内 OneHotEncoder(handle_unknown='ignore')。

数值列：DayOfWeek、ScheduledHour、ScheduledMinuteOfDay、IsWeekend、Distance、CRSElapsedTime。Logistic Regression 的数值标准化只在训练集拟合。ScheduledHour 与 ScheduledMinuteOfDay 存在相关性，第一版可保留，后续是否简化由训练/验证数据决定，不根据测试成绩挑选。

当前输出特征没有缺失值，没有做全局均值填补、标准化或类别编码。未来若新增字段需要这些操作，也应放在训练 Pipeline 内。月份保留用于描述统计，第一版未放入模型，避免把只在测试期出现的 3 月当成已经学过的月份。

## 数据划分

训练：2025-01-01 至 2025-02-14；验证：2025-02-15 至 2025-02-28；测试：2025-03-01 至 2025-03-31。按源文件的当地航班日期划分，不是全球统一 UTC 时点。同日不跨集合，未随机打散后重新划分。

用验证集比较模型、超参数及风险阈值；测试集留作最终评价。逻辑回归和随机森林必须使用相同划分。建议至少与“全部预测不延误”的简单基线比较；不要只报告 Accuracy，补充 Precision、Recall、F1、混淆矩阵等。没有保证模型能达到某个分数，也没有声称增加资源能实际减少多少延误。

## B 指定的八字段如何满足

FlightDate、Origin、Dest、CRSDepTime、DepTime、DepDelay、Cancelled 原样保留其数值或日期含义；Airline 明确由官方 Reporting_Airline 重命名。出发时间指登机口出发，不是 WheelsOff。DepTime、DepDelay、Cancelled 是分析/筛选字段，不能当作事前预测特征。取消航班在分析版保留，标签为空，模型版排除；备降航班若出发标签有效则保留。

## 交给 E / F

保留 metadata 的 record_id，把预测概率与原航班连接。风险等级阈值属于 B/E 的验证阶段决策，A 没有虚构高、中、低阈值。运营情景必须明确资源上限、干预假设和评价口径；不能把分类准确率直接解释成实际减少延误的效果。

本地保留了完整时期的描述统计和标签用于可复核交接。这不意味着可在模型选择时反复使用测试结果。模型预测界面输入计划字段，不能要求用户输入尚未发生的实际出发时间或延误分钟数。
