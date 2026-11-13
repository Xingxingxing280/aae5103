from __future__ import annotations

import pandas as pd


def scheduled_fields_from_crs(crs_dep_time: int, day_of_week: int) -> dict:
    """由计划 HHMM 与星期生成模型数值特征。2400 映射为当天 00:00。"""
    value = int(crs_dep_time)
    if value == 2400:
        hour, minute = 0, 0
    else:
        hour, minute = divmod(value, 100)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"非法计划出发时间：{crs_dep_time}")
    return {
        "DayOfWeek": int(day_of_week),
        "ScheduledHour": hour,
        "ScheduledMinuteOfDay": hour * 60 + minute,
        "IsWeekend": int(day_of_week in (6, 7)),
    }


def build_feature_row(
    *,
    airline: str,
    origin: str,
    dest: str,
    day_of_week: int,
    crs_dep_time: int,
    distance: float,
    crs_elapsed_time: float,
) -> pd.DataFrame:
    derived = scheduled_fields_from_crs(crs_dep_time, day_of_week)
    row = {
        "Airline": airline,
        "Origin": origin,
        "Dest": dest,
        **derived,
        "Distance": float(distance),
        "CRSElapsedTime": float(crs_elapsed_time),
    }
    return pd.DataFrame([row])
