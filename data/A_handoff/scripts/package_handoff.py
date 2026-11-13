"""Create a portable handoff; original large archives remain in data/raw."""
from pathlib import Path
import zipfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "deliverables"
out.mkdir(exist_ok=True)
target = out / "A_handoff.zip"
files = [ROOT / n for n in ["README.md","main.py","config.json","requirements.txt","requirements-run.txt"]]
for folder in ["scripts","docs","reports","notebooks","data/processed"]:
    files += [p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
# Small original metadata accompanies the package; large original ZIPs can be redownloaded.
files += list((ROOT / "data/raw").glob("*.json"))
with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(files):
        z.write(p,str(p.relative_to(ROOT)))
    z.writestr("交付包说明.txt", "此包包含处理后的真实CSV、代码、说明和检查记录。完整原始ZIP保留在项目data/raw目录，未重复装入压缩包。只使用模型数据时无需重下载；需要复核原始行时运行 python main.py --download 从BTS官方重下载并校验。\n")
with zipfile.ZipFile(target) as z:
    if z.testzip():
        raise ValueError("Delivery ZIP CRC error")
digest = hashlib.sha256(target.read_bytes()).hexdigest()
(out / "A_handoff.zip.sha256").write_text(f"{digest}  A_handoff.zip\n",encoding="ascii")
print(f"Created {target}: {target.stat().st_size:,} bytes, {len(files)} files; ZIP CRC passed. SHA256 {digest}")
