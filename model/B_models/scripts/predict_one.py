from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import artifacts_exist, load_artifacts, predict_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="用计划信息预测出发延误风险")
    parser.add_argument("--airline", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--day-of-week", type=int, required=True)
    parser.add_argument("--crs-dep-time", type=int, required=True, help="当地 HHMM，例如 1400")
    parser.add_argument("--distance", type=float, required=True, help="英里")
    parser.add_argument("--crs-elapsed-time", type=float, required=True, help="分钟")
    args = parser.parse_args()
    if not artifacts_exist():
        raise SystemExit("请先运行 python scripts/train_models.py")
    result = predict_schedule(
        load_artifacts(),
        airline=args.airline.upper(),
        origin=args.origin.upper(),
        dest=args.dest.upper(),
        day_of_week=args.day_of_week,
        crs_dep_time=args.crs_dep_time,
        distance=args.distance,
        crs_elapsed_time=args.crs_elapsed_time,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
