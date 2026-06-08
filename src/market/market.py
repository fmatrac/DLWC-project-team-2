import json
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Iterable

import yfinance as yf


OUTPUT_PATH = Path("data/raw/raw_market.jsonl")
DEFAULT_LOOKBACK_DAYS = 31
INTERVAL = "1d"


DEFAULT_TICKERS = [
    "^GSPC",
    "^DJI",
    "^IXIC",
    "^VIX",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "GS",
    "BAC",
    "XOM",
    "CVX",
]


def fetch_one(ticker: str, start: date, end: date) -> list[dict]:
    df = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=end.isoformat(),
        interval=INTERVAL,
        auto_adjust=False,
    )
    if df.empty:
        return []

    records = []
    for idx, row in df.iterrows():
        records.append({
            "ticker": ticker,
            "date": idx.date().isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "adj_close": float(row["Adj Close"]) if "Adj Close" in row else None,
            "volume": int(row["Volume"]) if row["Volume"] == row["Volume"] else 0,
        })
    return records


def fetch(
    start: date | str | None = None,
    end: date | str | None = None,
    tickers: Iterable[str] | None = None,
    output_path: Path | str = OUTPUT_PATH,
) -> Path:
    """
    Fetch market data for the given date range.

    Args:
        start: Start date (inclusive), either date or 'YYYY-MM-DD'.
        end: End date (exclusive), either date or 'YYYY-MM-DD'.
        tickers: Iterable of ticker symbols. If None, uses DEFAULT_TICKERS.
        output_path: Output JSONL path.

    Returns:
        Path to the written JSONL file.
    """
    if end is None:
        end = datetime.now(timezone.utc).date()
    elif isinstance(end, str):
        end = date.fromisoformat(end)

    if start is None:
        start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    elif isinstance(start, str):
        start = date.fromisoformat(start)

    if start >= end:
        raise ValueError(f"'start' must be earlier than 'end' (got start={start}, end={end})")

    if tickers is None:
        tickers = DEFAULT_TICKERS
    else:
        tickers = list(tickers)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"")

    total_rows = 0
    succeeded = 0

    with output_path.open("a", encoding="utf-8") as f:
        for ticker in tickers:
            try:
                records = fetch_one(ticker, start, end)
            except Exception as e:
                print(f"[market] skipped {ticker}: {e}")
                continue

            if not records:
                print(f"[market] empty {ticker}")
                continue

            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            total_rows += len(records)
            succeeded += 1

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(
        f"[market] range {start} -> {end} | "
        f"{succeeded}/{len(tickers)} tickers, {total_rows} rows, "
        f"{size_mb:.1f} MB -> {output_path}"
    )
    return output_path