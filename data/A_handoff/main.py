"""Entry point for Member A's data workflow (not a model training program)."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--download", action="store_true", help="Download/verify original BTS archives first")
args = parser.parse_args()
if args.download:
    subprocess.run([sys.executable, str(ROOT / "scripts/download_bts.py")], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py")], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/verify_data.py")], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/build_handoff.py")], check=True)
