
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


# ──────────────────────────────────────────────────────────────
# PARSERS
# ──────────────────────────────────────────────────────────────

def _load_embeddings(path: Path, field: str = "embeddinggemma_vec") -> dict[str, np.ndarray]:
    """
    Reads the embedded JSONL produced by embeder.py.
    Multiple records on the same date are mean-averaged into one vector.
    Returns: {date_str -> np.ndarray float32}
    """
    by_date: dict[str, list[np.ndarray]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            date = rec.get("date")
            vec  = rec.get(field)
            if date and vec:
                by_date[date].append(np.array(vec, dtype=np.float32))

    return {
        date: np.stack(vecs).mean(axis=0)
        for date, vecs in by_date.items()
    }


def _load_market(path: Path, ticker: str = "^GSPC") -> list[dict]:
    """
    Reads the market JSONL produced by market.py.
    Filters to a single ticker and returns rows sorted by date.
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("ticker") == ticker:
                rows.append(rec)

    rows.sort(key=lambda r: r["date"])
    return rows


# ──────────────────────────────────────────────────────────────
# SAMPLE BUILDER
# ──────────────────────────────────────────────────────────────

def _build_samples(
    market_rows: list[dict],
    embeddings:  dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Pairs news embedding on day t → S&P return on day t+1 (next trading day).

    Weekend/holiday news is rolled forward: if embeddings exist for
    non-trading days between two market rows (e.g. Sat/Sun between
    Friday and Monday), they are mean-averaged with any Friday embedding
    and assigned to the Friday→Monday pair.
    """
    trading_dates = [r["date"] for r in market_rows]
    trading_date_set = set(trading_dates)

    # Build a map: trading_date → list of embedding dates to aggregate
    # Each trading day t collects: its own embedding + any non-trading
    # days that fall between t and the previous trading day.
    from collections import defaultdict
    import numpy as np
    from datetime import date, timedelta

    date_to_emb_dates: dict[str, list[str]] = defaultdict(list)

    all_emb_dates = sorted(embeddings.keys())
    for emb_date in all_emb_dates:
        if emb_date in trading_date_set:
            date_to_emb_dates[emb_date].append(emb_date)
        else:
            # Find the next trading day for this embedding date
            d = date.fromisoformat(emb_date)
            for _ in range(7):          # look up to 7 days forward
                d += timedelta(days=1)
                candidate = d.isoformat()
                if candidate in trading_date_set:
                    date_to_emb_dates[candidate].append(emb_date)
                    break

    # Build aggregated embedding per trading day
    agg_embeddings: dict[str, np.ndarray] = {}
    for trading_date, emb_dates in date_to_emb_dates.items():
        vecs = [embeddings[ed] for ed in emb_dates if ed in embeddings]
        if vecs:
            agg_embeddings[trading_date] = np.stack(vecs).mean(axis=0)

    # Now build samples: feature from trading day t, label from t+1
    X_list, y_list, meta = [], [], []

    for i in range(len(market_rows) - 1):
        today = market_rows[i]
        nxt   = market_rows[i + 1]

        emb = agg_embeddings.get(today["date"])
        if emb is None:
            continue

        open_     = float(nxt["open"])
        adj_close = float(nxt["adj_close"])
        if open_ == 0:
            continue

        label = (adj_close - open_) / open_

        X_list.append(emb)
        y_list.append(label)
        meta.append({
            "news_date":        today["date"],
            "target_date":      nxt["date"],
            "emb_dates_used":   date_to_emb_dates[today["date"]],
        })

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, meta


# ──────────────────────────────────────────────────────────────
# NORMALIZER
# ──────────────────────────────────────────────────────────────

class FeatureNormalizer:
    """Fit on training split only, then apply to val/test."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        self.mean = X.mean(axis=0, keepdims=True)
        self.std  = X.std(axis=0,  keepdims=True)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ──────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ──────────────────────────────────────────────────────────────

class MarketNewsDataset(Dataset):
    """
    Each sample: {
        "x": FloatTensor (emb_dim,),   # mean-averaged daily news embedding
        "y": FloatTensor scalar,        # next-day S&P 500 intraday return
    }
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, meta: list[dict]):
        self.X    = torch.from_numpy(X)
        self.y    = torch.from_numpy(y)
        self.meta = meta

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": self.X[idx],
            "y": self.y[idx],
        }

    @property
    def emb_dim(self) -> int:
        return self.X.shape[1]


# ──────────────────────────────────────────────────────────────
# TIME-BASED SPLIT
# ──────────────────────────────────────────────────────────────

def _time_split(
    meta:        list[dict],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """
    Splits sample indices by chronological target_date.
    Never random — train always precedes val, val always precedes test.
    """
    dates = sorted({m["target_date"] for m in meta})
    n  = len(dates)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))

    train_dates = set(dates[:t1])
    val_dates   = set(dates[t1:t2])

    train_idx, val_idx, test_idx = [], [], []
    for i, m in enumerate(meta):
        d = m["target_date"]
        if d in train_dates:
            train_idx.append(i)
        elif d in val_dates:
            val_idx.append(i)
        else:
            test_idx.append(i)

    return train_idx, val_idx, test_idx


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def build_dataloaders(
    market_path:    Path | str,
    embedding_path: Path | str,
    ticker:         str   = "^GSPC",
    embedding_field: str  = "embeddinggemma_vec",
    batch_size:     int   = 64,
    train_ratio:    float = 0.70,
    val_ratio:      float = 0.15,
    normalize:      bool  = True,
    num_workers:    int   = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Full pipeline: load → join → split → normalize → DataLoaders.

    Args:
        market_path:     Path to raw_market.jsonl  (from market.py)
        embedding_path:  Path to embeded.jsonl      (from embeder.py)
        ticker:          Which ticker to use as the prediction target
        embedding_field: Key name for the vector in embeded.jsonl
        batch_size:      Samples per batch
        train_ratio:     Fraction of dates used for training
        val_ratio:       Fraction of dates used for validation
        normalize:       Whether to z-score the embedding features
        num_workers:     DataLoader worker processes

    Returns:
        train_loader, val_loader, test_loader,
        info {emb_dim, normalizer, meta, n_train, n_val, n_test}
    """
    market_path    = Path(market_path)
    embedding_path = Path(embedding_path)

    # 1. Load
    embeddings  = _load_embeddings(embedding_path, field=embedding_field)
    market_rows = _load_market(market_path, ticker=ticker)

    # 2. Join (only dates covered by GDELT)
    X, y, meta = _build_samples(market_rows, embeddings)

    if len(y) == 0:
        raise ValueError(
            "No overlapping dates found between market and embedding files. "
            "Check that both files cover the same date range."
        )

    print(f"[dataset] {len(y)} samples | emb_dim={X.shape[1]}")
    print(f"[dataset] date range: {meta[0]['target_date']} → {meta[-1]['target_date']}")

    # 3. Time-based split
    train_idx, val_idx, test_idx = _time_split(meta, train_ratio, val_ratio)
    print(f"[dataset] split → train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # 4. Normalize (fit only on train)
    normalizer = None
    if normalize and len(train_idx) > 0:
        normalizer = FeatureNormalizer()
        X[train_idx] = normalizer.fit_transform(X[train_idx])
        if val_idx:
            X[val_idx]  = normalizer.transform(X[val_idx])
        if test_idx:
            X[test_idx] = normalizer.transform(X[test_idx])

    # 5. Build dataset + loaders
    full_ds = MarketNewsDataset(X, y, meta)

    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(full_ds, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(full_ds, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    info = {
        "emb_dim":    full_ds.emb_dim,
        "normalizer": normalizer,
        "meta":       meta,
        "n_train":    len(train_idx),
        "n_val":      len(val_idx),
        "n_test":     len(test_idx),
    }
    return train_loader, val_loader, test_loader, info