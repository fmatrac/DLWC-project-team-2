import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent / "src" / "scraper"))
sys.path.append(str(Path(__file__).parent / "src" / "fetcher"))
sys.path.append(str(Path(__file__).parent / "src" / "market"))
sys.path.append(str(Path(__file__).parent / "src" / "embeder"))
sys.path.append(str(Path(__file__).parent / "src" / "data"))
sys.path.append(str(Path(__file__).parent / "src" / "model"))
sys.path.append(str(Path(__file__).parent / "src" / "training"))

import scraper
import fetcher
import market
import embeder
from data import build_dataloaders
from model import MarketCNN
from training import train, validate

INPUT_FILE  = Path("data/raw/raw_gdelt.csv")
OUTPUT_FILE = Path("data/processed/scraped_articles.jsonl")
FAILED_FILE = Path("data/processed/failed_urls.jsonl")
MARKET_FILE = Path("data/raw/raw_market.jsonl")
EMBEDED_FILE = Path("data/processed/embeded.jsonl")



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # files = fetcher.fetch_past_30_days()
    # fetcher.fetch(output_path=INPUT_FILE)
    # market.fetch(output_path=MARKET_FILE)
    
    # RAW_DIR = Path("data/raw")
    # def iter_daily_files(raw_dir: Path = RAW_DIR):
    #     yield from sorted(raw_dir.glob("raw_gdelt_*.csv"))
    
    # for path in iter_daily_files():
    #     scraper.run(path, OUTPUT_FILE, FAILED_FILE)
    
    # embeder.embed_jsonl_file(
    # OUTPUT_FILE,
    # EMBEDED_FILE,
    # embedding_field="embeddinggemma_vec",
    # batch_size=64,
    # )



    train_loader, val_loader, test_loader, info = build_dataloaders(
        market_path    = "data/raw/raw_market.jsonl",
        embedding_path = "data/processed/embeded.jsonl",
    )
    model = MarketCNN().to(device)

    train(model, train_loader, device=device)

    validate(model, val_loader, device)
    


if __name__ == "__main__":
    main()