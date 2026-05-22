import io
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import requests


OUTPUT_DIR = Path("data/raw")
TIMEOUT = 30
SNAPSHOT_INTERVAL_MINUTES = 15
DAYS_BACK = 30


def build_gdelt_url(dt: datetime) -> str:
    timestamp = dt.strftime("%Y%m%d%H%M%S")
    return f"http://data.gdeltproject.org/gdeltv2/{timestamp}.export.CSV.zip"


def generate_snapshots(date: datetime) -> list[datetime]:
    snapshots = []
    current = datetime(date.year, date.month, date.day, 0, 0, 0)
    end = datetime(date.year, date.month, date.day, 23, 45, 0)

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
        print(f"[fetcher] skipped {dt.strftime('%Y-%m-%d %H:%M')}: {e}")
        return False

    content_type = resp.headers.get("Content-Type", "")
    if "zip" not in content_type and resp.content[:4] != b"PK\x03\x04":
        print(f"[fetcher] skipped {dt.strftime('%Y-%m-%d %H:%M')}: not a zip")
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            first_file = zf.namelist()[0]
            with zf.open(first_file) as src, open(output_path, "ab") as dst:
                shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile:
        print(f"[fetcher] skipped {dt.strftime('%Y-%m-%d %H:%M')}: bad zip")
        return False

    return True


def daily_output_path(date: datetime, output_dir: Path = OUTPUT_DIR) -> Path:
    return output_dir / f"raw_gdelt_{date.strftime('%Y-%m-%d')}.csv"


def fetch_day(date: datetime, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = daily_output_path(date, output_dir)

    output_path.write_bytes(b"")
    snapshots = generate_snapshots(date)

    succeeded = 0
    for dt in snapshots:
        if fetch_one(dt, output_path):
            succeeded += 1

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(
        f"[fetcher] {date.strftime('%Y-%m-%d')} -> "
        f"{succeeded}/{len(snapshots)} snapshots, "
        f"{size_mb:.1f} MB -> {output_path}"
    )
    return output_path


def fetch_past_30_days(
    days_back: int = DAYS_BACK,
    end_date: datetime | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    if end_date is None:
        end_date = datetime.utcnow() - timedelta(days=1)

    created_files = []

    for i in range(days_back):
        day = end_date - timedelta(days=i)
        day = datetime(day.year, day.month, day.day)
        path = fetch_day(day, output_dir)
        created_files.append(path)

    return created_files