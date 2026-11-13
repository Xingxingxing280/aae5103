"""Download original monthly BTS Reporting Carrier archives; no third-party mirrors."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def download(year, month):
    name = f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    url = f"https://www.transtats.bts.gov/PREZIP/{name}"
    folder = ROOT / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    metadata = folder / (name + ".json")
    if dest.exists() and metadata.exists():
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if hashlib.sha256(dest.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Cached archive hash mismatch: {name}")
        print(f"Verified cached {name}", flush=True)
        return
    for attempt in range(3):
        part = dest.with_suffix(".zip.part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AAE5103-academic-data-preparation/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
                headers = {k: response.headers.get(k) for k in ("Content-Length", "Last-Modified", "ETag", "Content-Type")}
                total = 0
                while block := response.read(1024 * 1024):
                    out.write(block)
                    total += len(block)
            if headers["Content-Length"] and total != int(headers["Content-Length"]):
                raise ValueError("Incomplete response")
            with zipfile.ZipFile(part) as archive:
                bad = archive.testzip()
                if bad:
                    raise ValueError(f"ZIP CRC failed: {bad}")
                members = archive.namelist()
            digest = hashlib.sha256(part.read_bytes()).hexdigest()
            part.replace(dest)
            info = {"url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "bytes": total, "sha256": digest, "headers": headers, "members": members}
            metadata.write_text(json.dumps(info, indent=2), encoding="utf-8")
            print(f"Downloaded {name}: {total:,} bytes; SHA256 {digest}", flush=True)
            return
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda m: download(config["year"], m), config["months"]))
