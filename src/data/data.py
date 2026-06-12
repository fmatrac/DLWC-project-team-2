import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────
# PARSERS
# ──────────────────────────────────────────────────────────────

def _load_embeddings(
    path: Path,
    field: str = "embeddinggemma_vec",
    weight_field: str = "num_mentions",
) -> dict[str, np.ndarray]:
    """
    Reads embedded JSONL.

    Multiple records on the same date are combined into one vector using
    weighted mean with `weight_field`.

    Returns:
        {date_str -> np.ndarray float32}
    """
    by_date_vecs: dict[str, list[np.ndarray]] = defaultdict(list)
    by_date_weights: dict[str, list[float]] = defaultdict(list)

    skipped_no_date = 0
    skipped_no_vec = 0
    found_fields: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            found_fields.update(rec.keys())

            date = rec.get("date", "")
            if not date:
                skipped_no_date += 1
                continue
            date = str(date).replace("/", "-")[:10]

            vec = rec.get(field)
            if vec is None:
                skipped_no_vec += 1
                continue

            weight = rec.get(weight_field, 1.0)
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0
            if weight <= 0:
                weight = 1.0

            by_date_vecs[date].append(np.array(vec, dtype=np.float32))
            by_date_weights[date].append(weight)

    if skipped_no_date or skipped_no_vec:
        print(
            f"[embeddings] skipped {skipped_no_date} records missing 'date', "
            f"{skipped_no_vec} records missing field '{field}'"
        )
    if skipped_no_vec > 0:
        print(f"[embeddings] available fields in file: {sorted(found_fields)}")

    if not by_date_vecs:
        raise ValueError(
            f"No embedding records loaded from {path}. "
            f"Check that field '{field}' exists. "
            f"Fields found in file: {sorted(found_fields)}"
        )

    print(
        f"[embeddings] loaded {sum(len(v) for v in by_date_vecs.values())} vectors "
        f"across {len(by_date_vecs)} dates"
    )
    print(f"[embeddings] daily aggregation: weighted mean using '{weight_field}'")

    out = {}
    for date, vecs in by_date_vecs.items():
        weights = np.array(by_date_weights[date], dtype=np.float32)
        mat = np.stack(vecs).astype(np.float32)
        out[date] = np.average(mat, axis=0, weights=weights).astype(np.float32)

    return out


def _load_market(
    path: Path,
    tickers: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Reads market JSONL.

    If tickers is None, infer all tickers from the file in first-seen order.

    Returns:
        rows, resolved_tickers
    """
    all_rows = []
    discovered_tickers = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            ticker = rec.get("ticker")
            if not ticker:
                continue

            rec["date"] = str(rec.get("date", "")).replace("/", "-")[:10]
            all_rows.append(rec)
            discovered_tickers.append(ticker)

    if not all_rows:
        raise ValueError(f"No market rows found in {path}.")

    discovered_tickers = list(dict.fromkeys(discovered_tickers))

    if tickers is None:
        resolved_tickers = discovered_tickers
    else:
        resolved_tickers = list(dict.fromkeys(tickers))
        missing = [t for t in resolved_tickers if t not in set(discovered_tickers)]
        if missing:
            raise ValueError(
                f"Requested tickers not found in {path}: {missing}. "
                f"Tickers present in file: {discovered_tickers}"
            )

    ticker_set = set(resolved_tickers)
    rows = [r for r in all_rows if r["ticker"] in ticker_set]
    rows.sort(key=lambda r: (r["date"], r["ticker"]))

    print(
        f"[market] loaded {len(rows)} rows for {len(resolved_tickers)} tickers "
        f"across {len(sorted({r['date'] for r in rows}))} dates"
    )
    print(f"[market] tickers: {resolved_tickers}")

    return rows, resolved_tickers


# ──────────────────────────────────────────────────────────────
# SAMPLE BUILDER
# ──────────────────────────────────────────────────────────────

def _build_samples(
    market_rows: list[dict],
    embeddings: dict[str, np.ndarray],
    tickers: list[str],
    lookback_days: int = 4,
    target_mode: str = "return",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Builds aligned date samples.

    X(date) = mean of embedding vectors from [date-lookback_days, ..., date]
              for all dates available in embedding file.

    y(date) = vector over tickers for that market date.

    target_mode:
        - "return": (open - close) / open
        - "close": close
        - "open_close": [open, close] per ticker flattened
    """
    market_by_date: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in market_rows:
        market_by_date[row["date"]][row["ticker"]] = row

    market_dates = sorted(market_by_date.keys())
    overlap = set(market_dates) & set(embeddings.keys())

    print(f"[samples] same-day market↔embedding overlap: {len(overlap)} trading days")
    print(f"[samples] using lookback_days={lookback_days}")
    print(f"[samples] target_mode={target_mode}")

    X_list, y_list, meta = [], [], []

    skipped_no_emb_window = 0
    skipped_missing_tickers = 0
    skipped_bad_open = 0

    for d in market_dates:
        d_ts = np.datetime64(d)

        window_vecs = []
        window_dates = []
        for lag in range(lookback_days + 1):
            candidate = str(d_ts - np.timedelta64(lag, "D"))
            emb = embeddings.get(candidate)
            if emb is not None:
                window_vecs.append(emb)
                window_dates.append(candidate)

        if not window_vecs:
            skipped_no_emb_window += 1
            continue

        x_vec = np.stack(window_vecs).mean(axis=0).astype(np.float32)

        day_rows = market_by_date[d]
        if not all(t in day_rows for t in tickers):
            skipped_missing_tickers += 1
            continue

        y_vec = []
        bad_open = False

        for t in tickers:
            row = day_rows[t]
            open_ = float(row["open"])
            close_ = float(row["close"])

            if target_mode == "return":
                if open_ == 0:
                    bad_open = True
                    break
                y_vec.append((open_ - close_) / open_)

            elif target_mode == "close":
                y_vec.append(close_)

            elif target_mode == "open_close":
                y_vec.extend([open_, close_])

            else:
                raise ValueError(
                    f"Unknown target_mode='{target_mode}'. "
                    f"Use 'return', 'close', or 'open_close'."
                )

        if bad_open:
            skipped_bad_open += 1
            continue

        X_list.append(x_vec)
        y_list.append(np.array(y_vec, dtype=np.float32))
        meta.append({
            "date": d,
            "tickers": list(tickers),
            "embedding_window_dates": sorted(window_dates),
            "embedding_window_size": len(window_dates),
        })

    print(
        f"[samples] built {len(X_list)} samples "
        f"(skipped {skipped_no_emb_window} no-embedding-window, "
        f"{skipped_missing_tickers} missing-ticker-day, "
        f"{skipped_bad_open} bad-open)"
    )

    if not X_list:
        raise ValueError("No aligned samples could be built.")

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    return X, y, meta


# ──────────────────────────────────────────────────────────────
# NORMALIZER
# ──────────────────────────────────────────────────────────────

class FeatureNormalizer:
    """Fit on the whole X matrix and transform it."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────

class MarketNewsDataset(Dataset):
    """
    Each sample:
    {
        "x": FloatTensor (emb_dim,),
        "y": FloatTensor (out_dim,),
    }
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, meta: list[dict]):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.meta = meta

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": self.X[idx],
            "y": self.y[idx],
        }

    @property
    def emb_dim(self) -> int:
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        return self.y.shape[1]


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def build_dataloader(
    market_path: Path | str,
    embedding_path: Path | str,
    tickers: list[str] | None = None,
    embedding_field: str = "embeddinggemma_vec",
    embedding_weight_field: str = "num_mentions",
    lookback_days: int = 4,
    target_mode: str = "return",
    batch_size: int = 64,
    normalize: bool = True,
    shuffle: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, dict]:
    """
    Full pipeline:
        load embeddings + market data
        align by date
        create X and y
        wrap in Dataset and DataLoader

    Returns:
        loader, info
    """
    market_path = Path(market_path)
    embedding_path = Path(embedding_path)

    embeddings = _load_embeddings(
        embedding_path,
        field=embedding_field,
        weight_field=embedding_weight_field,
    )
    market_rows, resolved_tickers = _load_market(
        market_path,
        tickers=tickers,
    )

    X, y, meta = _build_samples(
        market_rows=market_rows,
        embeddings=embeddings,
        tickers=resolved_tickers,
        lookback_days=lookback_days,
        target_mode=target_mode,
    )

    normalizer = None
    if normalize:
        normalizer = FeatureNormalizer()
        X = normalizer.fit_transform(X)

    dataset = MarketNewsDataset(X, y, meta)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )

    info = {
        "emb_dim": dataset.emb_dim,
        "out_dim": dataset.out_dim,
        "tickers": resolved_tickers,
        "normalizer": normalizer,
        "meta": meta,
        "n_samples": len(dataset),
        "lookback_days": lookback_days,
        "target_mode": target_mode,
        "embedding_weight_field": embedding_weight_field,
    }

    print(
        f"[dataset] {len(dataset)} samples | "
        f"emb_dim={dataset.emb_dim} | out_dim={dataset.out_dim}"
    )
    print(f"[dataset] date range: {meta[0]['date']} → {meta[-1]['date']}")

    return loader, info