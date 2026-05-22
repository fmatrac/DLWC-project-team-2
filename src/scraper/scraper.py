import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
import trafilatura

MIN_TEXT_LEN = 300
MAX_WORKERS = 80
SLEEP_SECONDS = 0
MIN_MENTIONS = 3
MAX_URLS_PER_DAY = 100


def parse_gdelt_events(path):
    records = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 2:
                continue
            source_url = row[-1].strip()
            date_added = row[-2].strip() if len(row) >= 2 else None
            if not (source_url.startswith("http://") or source_url.startswith("https://")):
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
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    meta = trafilatura.extract_metadata(downloaded)
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    return {
        "final_url": url,
        "domain": urlparse(url).netloc,
        "title": meta.title if meta and meta.title else None,
        "authors": [meta.author] if meta and meta.author else [],
        "published_date": meta.date if meta and meta.date else None,
        "top_image": None,
        "text": text.strip() if text else None,
        "status": "success" if text and len(text.strip()) >= MIN_TEXT_LEN else "too_short",
        "error": None if text else "empty_text",
    }


def process_url(url, linked_rows):
    """
    Scrape this URL once and return a SINGLE article record.
    We keep some aggregated info from linked_rows if you want it later.
    """
    time.sleep(SLEEP_SECONDS)
    try:
        article_data = scrape_article(url)
        if article_data is None:
            raise RuntimeError("fetch_failed")

        # Optional: aggregate metadata from all GDELT rows that pointed to this URL
        event_ids = [r["global_event_id"] for r in linked_rows]
        max_mentions = max(r["num_mentions"] for r in linked_rows)

        merged = {
            **article_data,
            "linked_event_ids": event_ids,     # you can drop this if you don't care
            "max_num_mentions": max_mentions,  # you can also drop this
        }

        return merged, None

    except Exception as e:
        fail = {
            "source_url": url,
            "status": "failed",
            "error": str(e),
            "linked_event_ids": [r["global_event_id"] for r in linked_rows],
        }
        return None, fail


def run(input_file, output_file, failed_file):
    gdelt_rows = parse_gdelt_events(input_file)

    grouped_by_url = {}
    for row in gdelt_rows:
        grouped_by_url.setdefault(row["source_url"], []).append(row)

    # still prioritize URLs by max num_mentions, but only keep MAX_URLS_PER_DAY unique URLs
    sorted_urls = sorted(
        grouped_by_url.items(),
        key=lambda kv: max(r["num_mentions"] for r in kv[1]),
        reverse=True,
    )[:MAX_URLS_PER_DAY]

    with open(output_file, "w", encoding="utf-8") as out_f, \
         open(failed_file, "w", encoding="utf-8") as fail_f:

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_url, url, linked_rows): url
                for url, linked_rows in sorted_urls
            }

            for i, future in enumerate(as_completed(futures), start=1):
                ok_record, fail_record = future.result()

                if ok_record:
                    # exactly ONE JSON line per URL
                    out_f.write(json.dumps(ok_record, ensure_ascii=False) + "\n")

                if fail_record:
                    fail_f.write(json.dumps(fail_record, ensure_ascii=False) + "\n")

                if i % 50 == 0:
                    out_f.flush()
                    fail_f.flush()

    ok_count   = sum(1 for l in open(output_file, encoding="utf-8") if l.strip())
    fail_count = sum(1 for l in open(failed_file, encoding="utf-8") if l.strip())
    size_mb    = Path(output_file).stat().st_size / 1024 / 1024
    url_count  = len(sorted_urls)
    print(
        f"[scraper] {url_count} URLs -> {ok_count} records / {fail_count} failed, "
        f"{size_mb:.1f} MB -> {output_file}"
    )