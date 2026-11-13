"""成员 B 入口：训练模型或启动本地演示。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="AAE5103 成员 B：延误预测模型")
    parser.add_argument(
        "command",
        nargs="?",
        default="train",
        choices=["train", "demo"],
        help="train 训练并评价；demo 启动网页演示",
    )
    args = parser.parse_args()
    script = ROOT / ("scripts/train_models.py" if args.command == "train" else "app/web_demo.py")
    raise SystemExit(subprocess.call([sys.executable, str(script)]))


if __name__ == "__main__":
    main()
