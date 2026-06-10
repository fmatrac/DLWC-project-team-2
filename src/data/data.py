# import json
# from collections import defaultdict
# from pathlib import Path
# from typing import Optional

# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader, Subset


# # ──────────────────────────────────────────────────────────────
# # PARSERS
# # ──────────────────────────────────────────────────────────────

# def _load_embeddings(path: Path, field: str = "embeddinggemma_vec") -> dict[str, np.ndarray]:
#     """
#     Reads the embedded JSONL produced by embeder.py.
#     Multiple records on the same date are mean-averaged into one vector.
#     Returns: {date_str -> np.ndarray float32}
#     """
#     by_date: dict[str, list[np.ndarray]] = defaultdict(list)

#     skipped_no_date = 0
#     skipped_no_vec = 0
#     found_fields: set[str] = set()

#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             rec = json.loads(line)

#             found_fields.update(rec.keys())

#             date = rec.get("date", "")
#             if not date:
#                 skipped_no_date += 1
#                 continue
#             date = str(date).replace("/", "-")[:10]

#             vec = rec.get(field)
#             if vec is None:
#                 skipped_no_vec += 1
#                 continue

#             by_date[date].append(np.array(vec, dtype=np.float32))

#     if skipped_no_date or skipped_no_vec:
#         print(
#             f"[embeddings] skipped {skipped_no_date} records missing 'date', "
#             f"{skipped_no_vec} records missing field '{field}'"
#         )
#     if skipped_no_vec > 0:
#         print(f"[embeddings] available fields in file: {sorted(found_fields)}")
#     if not by_date:
#         raise ValueError(
#             f"No embedding records loaded from {path}. "
#             f"Check that field '{field}' exists. "
#             f"Fields found in file: {sorted(found_fields)}"
#         )

#     print(
#         f"[embeddings] loaded {sum(len(v) for v in by_date.values())} vectors "
#         f"across {len(by_date)} dates"
#     )

#     return {
#         date: np.stack(vecs).mean(axis=0).astype(np.float32)
#         for date, vecs in by_date.items()
#     }


# def _load_market(path: Path, tickers: list[str] | None = None) -> tuple[list[dict], list[str]]:
#     """
#     Reads the market JSONL produced by market.py.

#     If tickers is None, infer all tickers from the file in first-seen order.
#     Returns:
#         rows, resolved_tickers
#     where rows are filtered to resolved_tickers and sorted by (date, ticker).
#     """
#     all_rows = []
#     discovered_tickers = []

#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue

#             rec = json.loads(line)
#             ticker = rec.get("ticker")
#             if not ticker:
#                 continue

#             rec["date"] = str(rec.get("date", "")).replace("/", "-")[:10]
#             all_rows.append(rec)
#             discovered_tickers.append(ticker)

#     if not all_rows:
#         raise ValueError(f"No market rows found in {path}.")

#     discovered_tickers = list(dict.fromkeys(discovered_tickers))

#     if tickers is None:
#         resolved_tickers = discovered_tickers
#     else:
#         resolved_tickers = list(dict.fromkeys(tickers))
#         missing = [t for t in resolved_tickers if t not in set(discovered_tickers)]
#         if missing:
#             raise ValueError(
#                 f"Requested tickers not found in {path}: {missing}. "
#                 f"Tickers present in file: {discovered_tickers}"
#             )

#     ticker_set = set(resolved_tickers)
#     rows = [r for r in all_rows if r["ticker"] in ticker_set]
#     rows.sort(key=lambda r: (r["date"], r["ticker"]))

#     print(
#         f"[market] loaded {len(rows)} rows for {len(resolved_tickers)} tickers "
#         f"across {len(sorted({r['date'] for r in rows}))} dates"
#     )
#     print(f"[market] tickers: {resolved_tickers}")

#     return rows, resolved_tickers


# # ──────────────────────────────────────────────────────────────
# # SAMPLE BUILDER
# # ──────────────────────────────────────────────────────────────

# def _build_samples(
#     market_rows: list[dict],
#     embeddings: dict[str, np.ndarray],
#     tickers: list[str],
# ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
#     """
#     Builds same-day samples:

#       X(date) = mean-averaged embedding vector for that date
#       y(date) = vector over tickers of (open - close) / open for that date

#     A date is kept only if:
#       - it has an embedding
#       - all requested tickers are present that day
#       - every ticker has non-zero open
#     """
#     market_by_date: dict[str, dict[str, dict]] = defaultdict(dict)
#     for row in market_rows:
#         market_by_date[row["date"]][row["ticker"]] = row

#     market_dates = sorted(market_by_date.keys())
#     market_date_set = set(market_dates)

#     overlap = market_date_set & set(embeddings.keys())
#     print(f"[samples] market↔embedding overlap: {len(overlap)} trading days")

#     if len(overlap) == 0:
#         emb_sample = sorted(embeddings.keys())[:3]
#         market_sample = market_dates[:3]
#         print(f"[samples] sample embedding dates : {emb_sample}")
#         print(f"[samples] sample market dates     : {market_sample}")
#         raise ValueError(
#             "No overlapping dates between market and embedding files after "
#             "date normalisation. Check date formats in both files."
#         )

#     X_list, y_list, meta = [], [], []
#     skipped_no_emb = 0
#     skipped_missing_tickers = 0
#     skipped_bad_open = 0

#     for d in market_dates:
#         emb = embeddings.get(d)
#         if emb is None:
#             skipped_no_emb += 1
#             continue

#         day_rows = market_by_date[d]
#         if not all(t in day_rows for t in tickers):
#             skipped_missing_tickers += 1
#             continue

#         y_vec = []
#         bad_open = False

#         for t in tickers:
#             row = day_rows[t]
#             open_ = float(row["open"])
#             close_ = float(row["close"])

#             if open_ == 0:
#                 bad_open = True
#                 break

#             y_vec.append((open_ - close_) / open_)

#         if bad_open:
#             skipped_bad_open += 1
#             continue

#         X_list.append(emb)
#         y_list.append(np.array(y_vec, dtype=np.float32))
#         meta.append({
#             "date": d,
#             "tickers": list(tickers),
#         })

#     print(
#         f"[samples] built {len(X_list)} samples "
#         f"(skipped {skipped_no_emb} no-embedding, "
#         f"{skipped_missing_tickers} missing-ticker-day, "
#         f"{skipped_bad_open} zero-open)"
#     )

#     if not X_list:
#         raise ValueError(
#             f"No samples could be built. "
#             f"Overlapping dates: {len(overlap)}. "
#             f"Skipped due to missing embedding: {skipped_no_emb}, "
#             f"missing tickers: {skipped_missing_tickers}, "
#             f"zero open price: {skipped_bad_open}."
#         )

#     X = np.stack(X_list).astype(np.float32)
#     y = np.stack(y_list).astype(np.float32)
#     return X, y, meta


# # ──────────────────────────────────────────────────────────────
# # NORMALIZER
# # ──────────────────────────────────────────────────────────────

# class FeatureNormalizer:
#     """Fit on training split only, then apply to test."""

#     def __init__(self):
#         self.mean: Optional[np.ndarray] = None
#         self.std: Optional[np.ndarray] = None

#     def fit(self, X: np.ndarray) -> "FeatureNormalizer":
#         self.mean = X.mean(axis=0, keepdims=True)
#         self.std = X.std(axis=0, keepdims=True)
#         self.std[self.std < 1e-8] = 1.0
#         return self

#     def transform(self, X: np.ndarray) -> np.ndarray:
#         return ((X - self.mean) / self.std).astype(np.float32)

#     def fit_transform(self, X: np.ndarray) -> np.ndarray:
#         return self.fit(X).transform(X)


# # ──────────────────────────────────────────────────────────────
# # PYTORCH DATASET
# # ──────────────────────────────────────────────────────────────

# class MarketNewsDataset(Dataset):
#     """
#     Each sample: {
#         "x": FloatTensor (emb_dim,),
#         "y": FloatTensor (n_tickers,),
#     }
#     """

#     def __init__(self, X: np.ndarray, y: np.ndarray, meta: list[dict]):
#         self.X = torch.from_numpy(X)
#         self.y = torch.from_numpy(y)
#         self.meta = meta

#     def __len__(self) -> int:
#         return len(self.y)

#     def __getitem__(self, idx: int) -> dict:
#         return {
#             "x": self.X[idx],
#             "y": self.y[idx],
#         }

#     @property
#     def emb_dim(self) -> int:
#         return self.X.shape[1]

#     @property
#     def out_dim(self) -> int:
#         return self.y.shape[1]


# # ──────────────────────────────────────────────────────────────
# # TIME-BASED SPLIT
# # ──────────────────────────────────────────────────────────────

# def _time_split(
#     meta: list[dict],
#     train_ratio: float = 0.80,
# ) -> tuple[list[int], list[int]]:
#     """
#     Splits sample indices by chronological date.
#     Never random — train always precedes test.
#     """
#     dates = sorted({m["date"] for m in meta})
#     n = len(dates)

#     if n < 2:
#         raise ValueError("Need at least 2 distinct dates for a train/test split.")

#     cut = int(n * train_ratio)
#     cut = max(1, min(cut, n - 1))

#     train_dates = set(dates[:cut])

#     train_idx, test_idx = [], []
#     for i, m in enumerate(meta):
#         if m["date"] in train_dates:
#             train_idx.append(i)
#         else:
#             test_idx.append(i)

#     return train_idx, test_idx


# # ──────────────────────────────────────────────────────────────
# # MAIN ENTRY POINT
# # ──────────────────────────────────────────────────────────────

# def build_dataloaders(
#     market_path: Path | str,
#     embedding_path: Path | str,
#     tickers: list[str] | None = None,
#     embedding_field: str = "embeddinggemma_vec",
#     batch_size: int = 64,
#     train_ratio: float = 0.80,
#     normalize: bool = True,
#     num_workers: int = 0,
# ) -> tuple[DataLoader, DataLoader, dict]:
#     """
#     Full pipeline: load → join → split → normalize → DataLoaders.

#     Args:
#         market_path: Path to raw_market.jsonl (from market.py)
#         embedding_path: Path to embeded.jsonl (from embeder.py)
#         tickers: List of tickers to include in Y; if None, infer all from raw_market.jsonl
#         embedding_field: Key name for the vector in embeded.jsonl
#         batch_size: Samples per batch
#         train_ratio: Fraction of dates used for training
#         normalize: Whether to z-score the embedding features
#         num_workers: DataLoader worker processes

#     Returns:
#         train_loader, test_loader,
#         info {emb_dim, out_dim, tickers, normalizer, meta, n_train, n_test}
#     """
#     market_path = Path(market_path)
#     embedding_path = Path(embedding_path)

#     embeddings = _load_embeddings(embedding_path, field=embedding_field)
#     market_rows, resolved_tickers = _load_market(market_path, tickers=tickers)

#     X, y, meta = _build_samples(
#         market_rows=market_rows,
#         embeddings=embeddings,
#         tickers=resolved_tickers,
#     )

#     print(f"[dataset] {len(y)} samples | emb_dim={X.shape[1]} | out_dim={y.shape[1]}")
#     print(f"[dataset] date range: {meta[0]['date']} → {meta[-1]['date']}")

#     train_idx, test_idx = _time_split(meta, train_ratio=train_ratio)
#     print(f"[dataset] split → train={len(train_idx)}  test={len(test_idx)}")

#     normalizer = None
#     if normalize and len(train_idx) > 0:
#         normalizer = FeatureNormalizer()
#         X[train_idx] = normalizer.fit_transform(X[train_idx])
#         if test_idx:
#             X[test_idx] = normalizer.transform(X[test_idx])

#     full_ds = MarketNewsDataset(X, y, meta)

#     train_loader = DataLoader(
#         Subset(full_ds, train_idx),
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         drop_last=False,
#     )
#     test_loader = DataLoader(
#         Subset(full_ds, test_idx),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         drop_last=False,
#     )

#     info = {
#         "emb_dim": full_ds.emb_dim,
#         "out_dim": full_ds.out_dim,
#         "tickers": resolved_tickers,
#         "normalizer": normalizer,
#         "meta": meta,
#         "n_train": len(train_idx),
#         "n_test": len(test_idx),
#     }
#     return train_loader, test_loader, info

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


def _load_embeddings(
    path: Path,
    field: str = "embeddinggemma_vec",
    weight_field: str = "num_mentions",
) -> dict[str, np.ndarray]:
    """
    Reads the embedded JSONL produced by embeder.py.

    Multiple records on the same date are combined into one vector using a
    weighted mean, where `weight_field` (default: num_mentions) is used as the
    weight. If the weight is missing / invalid / <= 0, weight=1.0 is used.

    Returns: {date_str -> np.ndarray float32}
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


def _load_market(path: Path, tickers: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """
    Reads the market JSONL produced by market.py.

    If tickers is None, infer all tickers from the file in first-seen order.
    Returns:
        rows, resolved_tickers
    where rows are filtered to resolved_tickers and sorted by (date, ticker).
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
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Builds lag-aware samples:

      X(date) = mean of daily embedding vectors from the current market date and
                up to `lookback_days` previous calendar days that exist in the
                embedding file
      y(date) = vector over tickers of (open - close) / open for that market date

    Example:
      lookback_days=4 means events from date, date-1, date-2, date-3, date-4
      can affect the market day.

    A date is kept only if:
      - it has at least one embedding in the [date-lookback_days, date] window
      - all requested tickers are present that day
      - every ticker has non-zero open
    """
    market_by_date: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in market_rows:
        market_by_date[row["date"]][row["ticker"]] = row

    market_dates = sorted(market_by_date.keys())
    market_date_set = set(market_dates)

    overlap = market_date_set & set(embeddings.keys())
    print(f"[samples] same-day market↔embedding overlap: {len(overlap)} trading days")
    print(f"[samples] using lookback_days={lookback_days}")

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

        emb = np.stack(window_vecs).mean(axis=0).astype(np.float32)

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

            if open_ == 0:
                bad_open = True
                break

            y_vec.append((open_ - close_) / open_)

        if bad_open:
            skipped_bad_open += 1
            continue

        X_list.append(emb)
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
        f"{skipped_bad_open} zero-open)"
    )

    if not X_list:
        raise ValueError(
            f"No samples could be built. "
            f"Same-day overlapping dates: {len(overlap)}. "
            f"Skipped due to missing embedding window: {skipped_no_emb_window}, "
            f"missing tickers: {skipped_missing_tickers}, "
            f"zero open price: {skipped_bad_open}."
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    return X, y, meta


# ──────────────────────────────────────────────────────────────
# NORMALIZER
# ──────────────────────────────────────────────────────────────


class FeatureNormalizer:
    """Fit on training split only, then apply to test."""

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
# PYTORCH DATASET
# ──────────────────────────────────────────────────────────────


class MarketNewsDataset(Dataset):
    """
    Each sample: {
        "x": FloatTensor (emb_dim,),
        "y": FloatTensor (n_tickers,),
    }
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, meta: list[dict]):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
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

    @property
    def out_dim(self) -> int:
        return self.y.shape[1]


# ──────────────────────────────────────────────────────────────
# TIME-BASED SPLIT
# ──────────────────────────────────────────────────────────────


def _time_split(
    meta: list[dict],
    train_ratio: float = 0.80,
) -> tuple[list[int], list[int]]:
    """
    Splits sample indices by chronological date.
    Never random — train always precedes test.
    """
    dates = sorted({m["date"] for m in meta})
    n = len(dates)

    if n < 2:
        raise ValueError("Need at least 2 distinct dates for a train/test split.")

    cut = int(n * train_ratio)
    cut = max(1, min(cut, n - 1))

    train_dates = set(dates[:cut])

    train_idx, test_idx = [], []
    for i, m in enumerate(meta):
        if m["date"] in train_dates:
            train_idx.append(i)
        else:
            test_idx.append(i)

    return train_idx, test_idx


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────


def build_dataloaders(
    market_path: Path | str,
    embedding_path: Path | str,
    tickers: list[str] | None = None,
    embedding_field: str = "embeddinggemma_vec",
    embedding_weight_field: str = "num_mentions",
    lookback_days: int = 4,
    batch_size: int = 64,
    train_ratio: float = 0.80,
    normalize: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Full pipeline: load → join → split → normalize → DataLoaders.

    Args:
        market_path: Path to raw_market.jsonl (from market.py)
        embedding_path: Path to embeded.jsonl (from embeder.py)
        tickers: List of tickers to include in Y; if None, infer all from raw_market.jsonl
        embedding_field: Key name for the vector in embeded.jsonl
        embedding_weight_field: Weight field for combining same-day records, default 'num_mentions'
        lookback_days: Number of previous calendar days whose events may affect the market day.
                       Example: 4 means [date-4, ..., date] are aggregated.
        batch_size: Samples per batch
        train_ratio: Fraction of dates used for training
        normalize: Whether to z-score the embedding features
        num_workers: DataLoader worker processes

    Returns:
        train_loader, test_loader,
        info {emb_dim, out_dim, tickers, normalizer, meta, n_train, n_test, lookback_days}
    """
    market_path = Path(market_path)
    embedding_path = Path(embedding_path)

    embeddings = _load_embeddings(
        embedding_path,
        field=embedding_field,
        weight_field=embedding_weight_field,
    )
    market_rows, resolved_tickers = _load_market(market_path, tickers=tickers)

    X, y, meta = _build_samples(
        market_rows=market_rows,
        embeddings=embeddings,
        tickers=resolved_tickers,
        lookback_days=lookback_days,
    )

    print(f"[dataset] {len(y)} samples | emb_dim={X.shape[1]} | out_dim={y.shape[1]}")
    print(f"[dataset] date range: {meta[0]['date']} → {meta[-1]['date']}")

    train_idx, test_idx = _time_split(meta, train_ratio=train_ratio)
    print(f"[dataset] split → train={len(train_idx)}  test={len(test_idx)}")

    normalizer = None
    if normalize and len(train_idx) > 0:
        normalizer = FeatureNormalizer()
        X[train_idx] = normalizer.fit_transform(X[train_idx])
        if test_idx:
            X[test_idx] = normalizer.transform(X[test_idx])

    full_ds = MarketNewsDataset(X, y, meta)

    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        Subset(full_ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    info = {
        "emb_dim": full_ds.emb_dim,
        "out_dim": full_ds.out_dim,
        "tickers": resolved_tickers,
        "normalizer": normalizer,
        "meta": meta,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "lookback_days": lookback_days,
        "embedding_weight_field": embedding_weight_field,
    }
    return train_loader, test_loader, info