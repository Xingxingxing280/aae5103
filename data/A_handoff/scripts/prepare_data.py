"""Reproducible BTS data preparation. Run from any directory with pandas installed."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FIELDS = ["Year", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
                 "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
                 "CRSDepTime", "DepTime", "DepDelay", "DepDel15", "Cancelled", "Diverted",
                 "Distance", "CRSElapsedTime"]
FEATURES = ["Airline", "Origin", "Dest", "DayOfWeek", "ScheduledHour",
            "ScheduledMinuteOfDay", "IsWeekend", "Distance", "CRSElapsedTime"]
KEY = ["FlightDate", "Airline", "FlightNumber", "Origin", "Dest", "CRSDepTime"]


def hhmm_to_minutes(values):
    """Local HHMM; 2400 maps to midnight, preserving date in the original field."""
    n = pd.to_numeric(values, errors="coerce")
    valid = n.notna() & n.eq(n.round()) & (((n >= 0) & (n < 2400) & (n % 100 < 60)) | n.eq(2400))
    minutes = ((n // 100) * 60 + n % 100).where(valid) % 1440
    return minutes.astype("Int64")


def derive(frame):
    d = frame.rename(columns={"Reporting_Airline": "Airline", "Flight_Number_Reporting_Airline": "FlightNumber"}).copy()
    for field in ("Airline", "Origin", "Dest", "FlightNumber"):
        d[field] = d[field].astype("string").str.strip().replace("", pd.NA)
    # Do not guess dates or silently repair unexpected formats.
    d["FlightDate"] = pd.to_datetime(d.FlightDate, format="%Y-%m-%d", errors="raise")
    numeric = ["Year", "Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "DepTime", "DepDelay",
               "DepDel15", "Cancelled", "Diverted", "Distance", "CRSElapsedTime"]
    for field in numeric:
        d[field] = pd.to_numeric(d[field], errors="raise")
    for field in ("Cancelled", "Diverted", "DepDel15"):
        if not d[field].dropna().isin([0, 1]).all():
            raise ValueError(f"Invalid binary flag: {field}")
    if d.Cancelled.isna().any() or d.Diverted.isna().any():
        raise ValueError("Cancellation/diversion flags must be known")
    if not (d.Year.eq(d.FlightDate.dt.year) & d.Month.eq(d.FlightDate.dt.month)
            & d.DayofMonth.eq(d.FlightDate.dt.day) & d.DayOfWeek.eq(d.FlightDate.dt.dayofweek + 1)).all():
        raise ValueError("Source calendar fields disagree with FlightDate")
    known = d.Cancelled.eq(0) & d.DepDelay.notna() & d.DepDel15.notna()
    if not d.loc[known, "DepDel15"].eq(d.loc[known, "DepDelay"].ge(15).astype(int)).all():
        raise ValueError("DepDel15 disagrees with DepDelay >= 15")
    # Labels for cancelled flights are deliberately undefined, even if delay is recorded.
    d["TargetDelayed15"] = d.DepDel15.where(d.Cancelled.eq(0)).astype("Int64")
    d["ScheduledMinuteOfDay"] = hhmm_to_minutes(d.CRSDepTime)
    d["ScheduledHour"] = (d.ScheduledMinuteOfDay // 60).astype("Int64")
    d["IsWeekend"] = d.DayOfWeek.isin([6, 7]).astype(int)
    d["ScheduledAt2400"] = d.CRSDepTime.eq(2400).astype(int)
    d["ActualTimeInvalid"] = (d.DepTime.notna() & hhmm_to_minutes(d.DepTime).isna()).astype(int)
    return d


def assign_split(dates, config):
    return pd.Series(np.select([dates.le(pd.Timestamp(config["train_end"])),
                                dates.le(pd.Timestamp(config["validation_end"]))],
                               ["train", "validation"], default="test"), index=dates.index)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def run(config):
    out = ROOT / "data/processed"
    reports = ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    frames, sources = [], []
    for month in config["months"]:
        name = f'On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{config["year"]}_{month}.zip'
        path = ROOT / "data/raw" / name
        metadata = json.loads(path.with_suffix(".zip.json").read_text(encoding="utf-8"))
        if sha(path) != metadata["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")
        count, kept, members = 0, 0, []
        with zipfile.ZipFile(path) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if len(csvs) != 1:
                raise ValueError(f"Expected one CSV: {csvs}")
            member = csvs[0]
            members.append(member)
            with z.open(member) as f:
                for chunk in pd.read_csv(f, usecols=SOURCE_FIELDS, dtype="string", chunksize=100000):
                    dates = pd.to_datetime(chunk.FlightDate, format="%Y-%m-%d", errors="raise")
                    if not (dates.dt.year.eq(config["year"]) & dates.dt.month.eq(month)).all():
                        raise ValueError("Archive contains dates outside the named month")
                    selected = chunk.loc[chunk.Origin.isin(config["origins"])].copy()
                    selected["SourceArchive"] = name
                    selected["SourceMember"] = member
                    selected["SourceRow"] = selected.index + 1
                    frames.append(selected)
                    count += len(chunk)
                    kept += len(selected)
        sources.append({**metadata, "csv_members": members, "total_rows": count, "selected_rows": kept})
        print(f"Read {name}: {count:,} total, {kept:,} selected", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    if raw.empty:
        raise ValueError("No selected flights")
    d = derive(raw)
    original = len(d)
    relevant = [c for c in d.columns if not c.startswith("Source")]
    d = d.drop_duplicates(subset=relevant).copy()
    duplicates = original - len(d)
    # Ambiguous flight identities must be investigated, never silently dropped.
    conflicts = d.duplicated(KEY, keep=False)
    if conflicts.any():
        d.loc[conflicts].to_csv(reports / "identity_conflicts.csv", index=False)
        raise ValueError("Conflicting flight identity; inspect reports/identity_conflicts.csv")
    def record_id(row):
        text = "|".join(str(row[k]) for k in KEY)
        return hashlib.sha256(text.encode()).hexdigest()[:24]
    d.insert(0, "record_id", d.apply(record_id, axis=1))
    if not d.record_id.is_unique:
        raise ValueError("Record ID collision")
    d = d.sort_values(KEY).reset_index(drop=True)
    d["split"] = assign_split(d.FlightDate, config)
    rules = [
        ("cancelled", d.Cancelled.eq(1)),
        ("missing_label", d.TargetDelayed15.isna()),
        ("missing_identity", d[["Airline", "Origin", "Dest", "FlightNumber"]].isna().any(axis=1)),
        ("invalid_scheduled_time", d.ScheduledMinuteOfDay.isna()),
        ("invalid_distance", d.Distance.isna() | d.Distance.le(0)),
        ("invalid_scheduled_duration", d.CRSElapsedTime.isna() | d.CRSElapsedTime.le(0)),
    ]
    d["ModelExclusionReason"] = ""
    stages = [{"step": "origin_filter", "before": sum(s["total_rows"] for s in sources),
               "excluded": sum(s["total_rows"] for s in sources) - original, "after": original},
              {"step": "identical_relevant_record_duplicates", "before": original,
               "excluded": duplicates, "after": len(d)}]
    for reason, bad in rules:
        remaining = d.ModelExclusionReason.eq("")
        exclude = remaining & bad
        d.loc[exclude, "ModelExclusionReason"] = reason
        stages.append({"step": reason, "before": int(remaining.sum()), "excluded": int(exclude.sum()),
                       "after": int((remaining & ~bad).sum())})
    d["ModelEligible"] = d.ModelExclusionReason.eq("")
    d.to_csv(out / "cleaned_flights.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    model = d.loc[d.ModelEligible].copy()
    model[["record_id", "FlightDate", "split"] + FEATURES + ["TargetDelayed15"]].to_csv(
        out / "model_data.csv", index=False, date_format="%Y-%m-%d")
    split_summary = []
    for split in ("train", "validation", "test"):
        part = model.loc[model.split.eq(split)]
        if part.empty or part.TargetDelayed15.nunique() != 2:
            raise ValueError(f"Split {split} must have both classes")
        part[FEATURES].to_csv(out / f"X_{split}.csv", index=False)
        part[["TargetDelayed15"]].to_csv(out / f"y_{split}.csv", index=False)
        part[["record_id", "FlightDate", "Airline", "FlightNumber", "Origin", "Dest", "CRSDepTime"]].to_csv(
            out / f"metadata_{split}.csv", index=False, date_format="%Y-%m-%d")
        split_summary.append({"split": split, "rows": len(part), "start": str(part.FlightDate.min().date()),
                              "end": str(part.FlightDate.max().date()),
                              "delayed": int(part.TargetDelayed15.sum()), "delay_rate": float(part.TargetDelayed15.mean())})
    for group, fields in {"airport": ["Origin"], "month": ["Month"], "weekday": ["DayOfWeek"],
                          "scheduled_hour": ["ScheduledHour"]}.items():
        # Descriptive tables cover the full study period, not model input features.
        summary = d.groupby(fields, dropna=False).agg(flights=("record_id", "size"),
                    cancellations=("Cancelled", "sum"), labelled_flights=("TargetDelayed15", "count"),
                    delayed_flights=("TargetDelayed15", "sum"), delay_rate=("TargetDelayed15", "mean"))
        summary.to_csv(reports / f"summary_{group}.csv", encoding="utf-8-sig")
    pd.DataFrame(stages).to_csv(reports / "cleaning_audit.csv", index=False)
    pd.DataFrame(split_summary).to_csv(reports / "split_summary.csv", index=False)
    pd.DataFrame({"missing": d.isna().sum(), "missing_fraction": d.isna().mean()}).to_csv(reports / "missingness.csv")
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                "features": FEATURES, "target": "TargetDelayed15", "sources": sources,
                "rows_cleaned": len(d), "rows_model": len(model), "splits": split_summary,
                "checks": {"calendar_consistency": "passed", "label_consistency": "passed",
                           "identity_conflicts": 0, "unique_ids": True,
                           "scheduled_2400_count": int(d.ScheduledAt2400.sum()),
                           "invalid_actual_time_count": int(d.ActualTimeInvalid.sum())},
                "script_sha256": sha(Path(__file__)),
                "output_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(out.glob("*.csv"))}}
    (reports / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rows_cleaned": len(d), "rows_model": len(model), "splits": split_summary}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = p.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")))
