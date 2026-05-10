import csv
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from newspaper import Article

INPUT_FILE = "data/raw_gdelt.csv"          # tab-delimited GDELT file
OUTPUT_FILE = "data/scraped_articles.jsonl"
FAILED_FILE = "data/failed_urls.jsonl"


SLEEP_SECONDS = 0.02
MIN_TEXT_LEN = 300

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
                "num_mentions": row[31].strip() if len(row) > 31 else None,
                "num_sources": row[32].strip() if len(row) > 32 else None,
                "num_articles": row[33].strip() if len(row) > 33 else None,
                "avg_tone": row[34].strip() if len(row) > 34 else None,
                "date_added": date_added,
                "source_url": source_url,
            })
    return records

def scrape_article(url, language="en"):
    article = Article(url, language=language)
    article.download()
    article.parse()

    text = article.text.strip() if article.text else None
    publish_date = article.publish_date.isoformat() if article.publish_date else None

    return {
        "final_url": article.url,
        "domain": urlparse(article.url).netloc if article.url else urlparse(url).netloc,
        "title": article.title.strip() if article.title else None,
        "authors": article.authors if article.authors else [],
        "published_date": publish_date,
        "top_image": article.top_image if article.top_image else None,
        "text": text,
        "status": "success" if text and len(text) >= MIN_TEXT_LEN else "too_short",
        "error": None if text else "empty_text"
    }

def main():
    Path("data").mkdir(exist_ok=True)

    gdelt_rows = parse_gdelt_events(INPUT_FILE)

    grouped_by_url = {}
    for row in gdelt_rows:
        grouped_by_url.setdefault(row["source_url"], []).append(row)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f, \
         open(FAILED_FILE, "w", encoding="utf-8") as fail_f:

        for i, (url, linked_rows) in enumerate(grouped_by_url.items(), start=1):
            try:
                article_data = scrape_article(url)

                for row in linked_rows:
                    merged = {**row, **article_data}
                    out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")

            except Exception as e:
                fail_f.write(json.dumps({
                    "source_url": url,
                    "status": "failed",
                    "error": str(e),
                    "linked_event_ids": [r["global_event_id"] for r in linked_rows]
                }, ensure_ascii=False) + "\n")

            time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()