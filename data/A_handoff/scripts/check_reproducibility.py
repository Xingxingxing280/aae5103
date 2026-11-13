"""Re-run preparation and compare every model/analysis CSV byte for byte."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
before = json.loads((ROOT / "reports/manifest.json").read_text())["output_sha256"]
subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py")], check=True)
after = json.loads((ROOT / "reports/manifest.json").read_text())["output_sha256"]
if before != after:
    raise ValueError("Non-deterministic CSV output")
subprocess.run([sys.executable, str(ROOT / "scripts/verify_data.py")], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/build_handoff.py")], check=True)
nb = json.loads((ROOT / "notebooks/01_data_handoff.ipynb").read_text(encoding="utf-8"))
context = {}
# Execute code cells as a smoke check; this is not a Jupyter render/visual check.
import os
os.chdir(ROOT)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        exec(compile("".join(cell["source"]), "handoff_notebook", "exec"), context)
result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "identical_csv_files":len(after),
          "csv_sha256_unchanged": True, "notebook_code_smoke_check": "passed (Python exec; not Jupyter rendering)",
          "output_sha256": after}
(ROOT / "reports/reproducibility.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(f"Reproducibility passed: {len(after)} CSV hashes unchanged. Notebook code cells executed successfully.")
