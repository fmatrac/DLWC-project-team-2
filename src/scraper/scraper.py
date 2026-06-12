import csv
import json
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from urllib.parse import urlparse

import trafilatura
from trafilatura.settings import DEFAULT_CONFIG


MIN_TEXT_LEN = 300
MAX_WORKERS = 16
SLEEP_SECONDS = 0
MIN_MENTIONS = 3
MAX_URLS_PER_DAY = 100
DOWNLOAD_TIMEOUT = 10
GLOBAL_BATCH_TIMEOUT = 30


def make_trafilatura_config():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["DEFAULT"]["DOWNLOAD_TIMEOUT"] = str(DOWNLOAD_TIMEOUT)
    cfg["DEFAULT"]["SLEEP_TIME"] = "0"
    cfg["DEFAULT"]["MAX_REDIRECTS"] = "2"
    return cfg


TRAFILATURA_CONFIG = make_trafilatura_config()


def parse_gdelt_events(path):
    records = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if not row or len(row) < 2:
                continue

            source_url = row[-1].strip()
            date_added = row[-2].strip() if len(row) >= 2 else None

            if source_url.startswith("[") and "](" in source_url and source_url.endswith(")"):
                try:
                    source_url = source_url.split("](", 1)[1][:-1].strip()
                except Exception:
                    pass

            if not source_url.startswith(("http://", "https://")):
                continue

            try:
                num_mentions = int(row[31].strip()) if len(row) > 31 else 0
            except ValueError:
                num_mentions = 0

            if num_mentions < MIN_MENTIONS:
                continue

            records.append({
                "global_event_id": row[0].strip() if len(row) > 0 else None,
                "day": row[1].strip() if len(row) > 1 else None,
                "month_year": row[2].strip() if len(row) > 2 else None,
                "year": row[3].strip() if len(row) > 3 else None,
                "fraction_date": row[4].strip() if len(row) > 4 else None,
                "event_code": row[26].strip() if len(row) > 26 else None,
                "event_base_code": row[27].strip() if len(row) > 27 else None,
                "event_root_code": row[28].strip() if len(row) > 28 else None,
                "quad_class": row[29].strip() if len(row) > 29 else None,
                "goldstein_scale": row[30].strip() if len(row) > 30 else None,
                "num_mentions": num_mentions,
                "num_sources": row[32].strip() if len(row) > 32 else None,
                "num_articles": row[33].strip() if len(row) > 33 else None,
                "avg_tone": row[34].strip() if len(row) > 34 else None,
                "date_added": date_added,
                "source_url": source_url,
            })
    return records


def scrape_article(url):
    downloaded = trafilatura.fetch_url(url, config=TRAFILATURA_CONFIG)
    if not downloaded:
        return None

    meta = trafilatura.extract_metadata(downloaded)
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        config=TRAFILATURA_CONFIG,
        fast=True,
    )

    cleaned = text.strip() if text else None

    return {
        "final_url": url,
        "domain": urlparse(url).netloc,
        "title": meta.title if meta and meta.title else None,
        "authors": [meta.author] if meta and meta.author else [],
        "published_date": meta.date if meta and meta.date else None,
        "top_image": None,
        "text": cleaned,
        "status": "success" if cleaned and len(cleaned) >= MIN_TEXT_LEN else "too_short",
        "error": None if cleaned else "empty_text",
    }


def process_url(url, linked_rows):
    time.sleep(SLEEP_SECONDS)
    article_data = scrape_article(url)
    if article_data is None:
        raise RuntimeError("fetch_failed")

    event_ids = [r["global_event_id"] for r in linked_rows]
    max_mentions = max(r["num_mentions"] for r in linked_rows)

    merged = {
        **article_data,
        "linked_event_ids": event_ids,
        "max_num_mentions": max_mentions,
    }
    return merged, None


def run(input_file, output_file, failed_file):
    gdelt_rows = parse_gdelt_events(input_file)

    grouped_by_url = {}
    for row in gdelt_rows:
        grouped_by_url.setdefault(row["source_url"], []).append(row)

    sorted_urls = sorted(
        grouped_by_url.items(),
        key=lambda kv: max(r["num_mentions"] for r in kv[1]),
        reverse=True,
    )[:MAX_URLS_PER_DAY]

    print(f"[run] parsed rows: {len(gdelt_rows)} | unique urls: {len(sorted_urls)}")

    out_f = open(output_file, "a", encoding="utf-8")
    fail_f = open(failed_file, "a", encoding="utf-8")

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = {
        executor.submit(process_url, url, linked_rows): (url, linked_rows)
        for url, linked_rows in sorted_urls
    }

    completed = 0
    batch_start = time.time()

    try:
        for future in as_completed(futures, timeout=GLOBAL_BATCH_TIMEOUT):
            url, linked_rows = futures[future]
            completed += 1

            try:
                ok_record, fail_record = future.result()
            except Exception as e:
                ok_record = None
                fail_record = {
                    "source_url": url,
                    "status": "failed",
                    "error": str(e),
                    "linked_event_ids": [r["global_event_id"] for r in linked_rows],
                }

            if ok_record:
                out_f.write(json.dumps(ok_record, ensure_ascii=False) + "\n")

            if fail_record:
                fail_f.write(json.dumps(fail_record, ensure_ascii=False) + "\n")

            if completed % 20 == 0:
                out_f.flush()
                fail_f.flush()
                elapsed = time.time() - batch_start
                print(f"[progress] done {completed}/{len(sorted_urls)} in {elapsed:.1f}s")

    except TimeoutError:
        print(f"[timeout] batch exceeded {GLOBAL_BATCH_TIMEOUT}s, forcing shutdown")

        for future, (url, linked_rows) in futures.items():
            if not future.done():
                fail_record = {
                    "source_url": url,
                    "status": "failed",
                    "error": f"batch_timeout_{GLOBAL_BATCH_TIMEOUT}s",
                    "linked_event_ids": [r["global_event_id"] for r in linked_rows],
                }
                fail_f.write(json.dumps(fail_record, ensure_ascii=False) + "\n")

    finally:
        out_f.flush()
        fail_f.flush()
        out_f.close()
        fail_f.close()
        executor.shutdown(wait=False, cancel_futures=True)

    ok_count = sum(1 for l in open(output_file, encoding="utf-8") if l.strip())
    fail_count = sum(1 for l in open(failed_file, encoding="utf-8") if l.strip())
    size_mb = Path(output_file).stat().st_size / 1024 / 1024 if Path(output_file).exists() else 0.0

    print(
        f"[scraper] {len(sorted_urls)} URLs -> {ok_count} records / {fail_count} failed, "
        f"{size_mb:.1f} MB -> {output_file}"
    )