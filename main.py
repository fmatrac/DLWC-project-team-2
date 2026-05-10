import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src" / "scraper"))
import scraper

INPUT_FILE  = Path("data/raw/raw_gdelt.csv")
OUTPUT_FILE = Path("data/processed/scraped_articles.jsonl")
FAILED_FILE = Path("data/processed/failed_urls.jsonl")


def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    scraper.run(INPUT_FILE, OUTPUT_FILE, FAILED_FILE)


if __name__ == "__main__":
    main()