import io
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import requests

OUTPUT_PATH = Path("data/raw/raw_gdelt.csv")
TIMEOUT = 30
SNAPSHOT_INTERVAL_MINUTES = 15


def build_gdelt_url(dt: datetime) -> str:
    timestamp = dt.strftime("%Y%m%d%H%M%S")
    return f"http://data.gdeltproject.org/gdeltv2/{timestamp}.export.CSV.zip"


def generate_snapshots(date: datetime) -> list:
    snapshots = []
    current = datetime(date.year, date.month, date.day, 0, 0, 0)
    end     = datetime(date.year, date.month, date.day, 23, 45, 0)
    while current <= end:
        snapshots.append(current)
        current += timedelta(minutes=SNAPSHOT_INTERVAL_MINUTES)
    return snapshots


def fetch_one(dt: datetime, output_path: Path) -> bool:
    url = build_gdelt_url(dt)
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[fetcher] skipped {dt.strftime('%H:%M')}: {e}")
        return False

    content_type = resp.headers.get("Content-Type", "")
    if "zip" not in content_type and not resp.content[:4] == b"PK\x03\x04":
        return False
    
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as src:
                with open(output_path, "ab") as dst:
                    shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile:
        print(f"[fetcher]: skipped {dt.strftime('%H%M')}: ZIP")
        return False
    return True


def fetch(date=None, output_path=OUTPUT_PATH) -> Path:
    if date is None:
        date = datetime.utcnow() - timedelta(days=1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"")

    snapshots = generate_snapshots(date)

    succeeded = 0
    for i, dt in enumerate(snapshots, start=1):
        ok = fetch_one(dt, output_path)
        if ok:
            succeeded += 1

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[fetcher] {succeeded}/{len(snapshots)} snapshots, "
          f"{size_mb:.1f} MB -> {output_path}")
    return output_path