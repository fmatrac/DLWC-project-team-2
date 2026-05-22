import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src" / "scraper"))
sys.path.append(str(Path(__file__).parent / "src" / "fetcher"))
sys.path.append(str(Path(__file__).parent / "src" / "market"))
sys.path.append(str(Path(__file__).parent / "src" / "embeder"))
import scraper
import fetcher
import market
import embeder

INPUT_FILE  = Path("data/raw/raw_gdelt.csv")
OUTPUT_FILE = Path("data/processed/scraped_articles.jsonl")
FAILED_FILE = Path("data/processed/failed_urls.jsonl")
MARKET_FILE = Path("data/raw/raw_market.jsonl")
EMBEDED_FILE = Path("data/processed/embeded.jsonl")



def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # fetcher.fetch(output_path=INPUT_FILE)
    # market.fetch(output_path=MARKET_FILE)
    # scraper.run(INPUT_FILE, OUTPUT_FILE, FAILED_FILE)

    embeder.embed_jsonl_file(
    OUTPUT_FILE,
    EMBEDED_FILE,
    embedding_field="embeddinggemma_vec",
    batch_size=64,
    )


if __name__ == "__main__":
    main()