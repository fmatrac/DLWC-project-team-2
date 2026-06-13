# # import numpy as np
# # from scipy import stats
# # from sklearn.decomposition import PCA
# # from sklearn.cross_decomposition import CCA


# # def pca_pearson(X: np.ndarray, y: np.ndarray, n_components: int = 10) -> dict:
# #     n_components = min(n_components, X.shape[0], X.shape[1])
# #     pca = PCA(n_components=n_components)
# #     Z = pca.fit_transform(X)  # (N, n_components)

# #     pearson_r = []
# #     p_values = []
# #     for i in range(n_components):
# #         r, p = stats.pearsonr(Z[:, i], y)
# #         pearson_r.append(r)
# #         p_values.append(p)

# #     return {
# #         "components": n_components,
# #         "pearson_r": pearson_r,
# #         "p_values": p_values,
# #         "explained_var": pca.explained_variance_ratio_.tolist(),
# #     }


# # def spearman_on_pca(X: np.ndarray, y: np.ndarray, n_components: int = 10) -> dict:
# #     n_components = min(n_components, X.shape[0], X.shape[1])
# #     pca = PCA(n_components=n_components)
# #     Z = pca.fit_transform(X)

# #     spearman_r = []
# #     p_values = []
# #     for i in range(n_components):
# #         r, p = stats.spearmanr(Z[:, i], y)
# #         spearman_r.append(r)
# #         p_values.append(p)

# #     return {
# #         "components": n_components,
# #         "spearman_r": spearman_r,
# #         "p_values": p_values,
# #         "explained_var": pca.explained_variance_ratio_.tolist(),
# #     }


# # def cca_correlation( X: np.ndarray, y: np.ndarray, n_components: int = 1, ) -> dict:
# #     y_col = y.reshape(-1, 1)

# #     n_pca = min(50, X.shape[0] - 1, X.shape[1])
# #     pca = PCA(n_components=n_pca)
# #     X_reduced = pca.fit_transform(X)

# #     n_components = min(n_components, n_pca)
# #     cca = CCA(n_components=n_components)
# #     cca.fit(X_reduced, y_col)

# #     X_c, y_c = cca.transform(X_reduced, y_col)

# #     canonical_corrs = []
# #     for i in range(n_components):
# #         r, p = stats.pearsonr(X_c[:, i], y_c[:, i])
# #         canonical_corrs.append({"r": r, "p": p})

# #     return {
# #         "n_components": n_components,
# #         "canonical_correlations": canonical_corrs,
# #         "pca_components_used": n_pca,
# #         "explained_var_by_pca": pca.explained_variance_ratio_.sum(),
# #     }


# # def rv_coefficient(X: np.ndarray, y: np.ndarray) -> dict:
# #     if y.ndim == 1:
# #         y = y.reshape(-1, 1)

# #     def _rv(A, B):
# #         SA = A @ A.T
# #         SB = B @ B.T
# #         num = np.trace(SA @ SB)
# #         denom = np.sqrt(np.trace(SA @ SA) * np.trace(SB @ SB))
# #         return num / denom if denom > 0 else 0.0

# #     observed_rv = _rv(X, y)

# #     n_permutations = 999
# #     count_extreme = 0
# #     rng = np.random.default_rng(42)
# #     for _ in range(n_permutations):
# #         y_perm = rng.permutation(y)
# #         if _rv(X, y_perm) >= observed_rv:
# #             count_extreme += 1

# #     p_value = (count_extreme + 1) / (n_permutations + 1)

# #     return {"rv": observed_rv, "p_value": p_value, "n_permutations": n_permutations}


# # def run_all(X: np.ndarray, y: np.ndarray, verbose: bool = True) -> dict:
# #     results = {}

# #     print("\nPCA + Pearson")
# #     r = pca_pearson(X, y)
# #     results["pca_pearson"] = r
# #     if verbose:
# #         for i in range(r["components"]):
# #             sig = "*" if r["p_values"][i] < 0.05 else ""
# #             print(f"  PC{i+1:2d}  r={r['pearson_r'][i]:+.3f}  p={r['p_values'][i]:.3f} expvar={r['explained_var'][i]:.3f}  {sig}")

# #     print("\nPCA + Spearman")
# #     r = spearman_on_pca(X, y)
# #     results["pca_spearman"] = r
# #     if verbose:
# #         for i in range(r["components"]):
# #             sig = "*" if r["p_values"][i] < 0.05 else ""
# #             print( f"  PC{i+1:2d}  ρ={r['spearman_r'][i]:+.3f}  p={r['p_values'][i]:.3f}  {sig}" )

# #     print("\nCCA")
# #     r = cca_correlation(X, y, n_components=1)
# #     results["cca"] = r
# #     if verbose:
# #         cc = r["canonical_correlations"][0]
# #         print(f"  Canonical correlation r={cc['r']:+.3f}  p={cc['p']:.3f}")
# #         print(f"  (PCA pre-reduction to {r['pca_components_used']} dims, explaining {r['explained_var_by_pca']:.1%} of variance)")

# #     print("\nRV Coefficient")
# #     r = rv_coefficient(X, y)
# #     results["rv"] = r
# #     if verbose:
# #         print(f"  RV={r['rv']:.4f}  p={r['p_value']:.3f} (permutation test, n={r['n_permutations']})")

# #     return results

# import numpy as np
# from scipy import stats
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import CCA


# def _ensure_2d_y(y: np.ndarray) -> np.ndarray:
#     y = np.asarray(y, dtype=np.float64)
#     if y.ndim == 1:
#         y = y.reshape(-1, 1)
#     return y


# def _default_ticker_names(n: int) -> list[str]:
#     return [f"T{i+1}" for i in range(n)]


# def pca_pearson(
#     X: np.ndarray,
#     y: np.ndarray,
#     n_components: int = 10,
#     ticker_names: list[str] | None = None,
# ) -> dict:
#     y = _ensure_2d_y(y)
#     n_components = min(n_components, X.shape[0], X.shape[1])

#     pca = PCA(n_components=n_components)
#     Z = pca.fit_transform(X)

#     if ticker_names is None:
#         ticker_names = _default_ticker_names(y.shape[1])

#     per_component = []

#     for i in range(n_components):
#         by_ticker = {}
#         for j, ticker in enumerate(ticker_names):
#             r, p = stats.pearsonr(Z[:, i], y[:, j])
#             by_ticker[ticker] = {"r": float(r), "p": float(p)}
#         per_component.append({
#             "component": i + 1,
#             "explained_var": float(pca.explained_variance_ratio_[i]),
#             "by_ticker": by_ticker,
#         })

#     return {
#         "components": n_components,
#         "targets": y.shape[1],
#         "results": per_component,
#     }


# def spearman_on_pca(
#     X: np.ndarray,
#     y: np.ndarray,
#     n_components: int = 10,
#     ticker_names: list[str] | None = None,
# ) -> dict:
#     y = _ensure_2d_y(y)
#     n_components = min(n_components, X.shape[0], X.shape[1])

#     pca = PCA(n_components=n_components)
#     Z = pca.fit_transform(X)

#     if ticker_names is None:
#         ticker_names = _default_ticker_names(y.shape[1])

#     per_component = []

#     for i in range(n_components):
#         by_ticker = {}
#         for j, ticker in enumerate(ticker_names):
#             r, p = stats.spearmanr(Z[:, i], y[:, j])
#             by_ticker[ticker] = {"r": float(r), "p": float(p)}
#         per_component.append({
#             "component": i + 1,
#             "explained_var": float(pca.explained_variance_ratio_[i]),
#             "by_ticker": by_ticker,
#         })

#     return {
#         "components": n_components,
#         "targets": y.shape[1],
#         "results": per_component,
#     }


# def cca_correlation(
#     X: np.ndarray,
#     y: np.ndarray,
#     n_components: int = 1,
# ) -> dict:
#     y = _ensure_2d_y(y)

#     n_pca = min(50, X.shape[0] - 1, X.shape[1])
#     pca = PCA(n_components=n_pca)
#     X_reduced = pca.fit_transform(X)

#     n_components = min(n_components, n_pca, y.shape[1])
#     cca = CCA(n_components=n_components)
#     cca.fit(X_reduced, y)

#     X_c, y_c = cca.transform(X_reduced, y)

#     canonical_corrs = []
#     for i in range(n_components):
#         r, p = stats.pearsonr(X_c[:, i], y_c[:, i])
#         canonical_corrs.append({"r": float(r), "p": float(p)})

#     return {
#         "n_components": n_components,
#         "canonical_correlations": canonical_corrs,
#         "pca_components_used": n_pca,
#         "explained_var_by_pca": float(pca.explained_variance_ratio_.sum()),
#     }


# def rv_coefficient(X: np.ndarray, y: np.ndarray) -> dict:
#     y = _ensure_2d_y(y)

#     def _rv(A, B):
#         SA = A @ A.T
#         SB = B @ B.T
#         num = np.trace(SA @ SB)
#         denom = np.sqrt(np.trace(SA @ SA) * np.trace(SB @ SB))
#         return num / denom if denom > 0 else 0.0

#     observed_rv = _rv(X, y)

#     n_permutations = 999
#     count_extreme = 0
#     rng = np.random.default_rng(42)

#     for _ in range(n_permutations):
#         y_perm = rng.permutation(y)
#         if _rv(X, y_perm) >= observed_rv:
#             count_extreme += 1

#     p_value = (count_extreme + 1) / (n_permutations + 1)

#     return {
#         "rv": float(observed_rv),
#         "p_value": float(p_value),
#         "n_permutations": n_permutations,
#     }


# def run_all(
#     X: np.ndarray,
#     y: np.ndarray,
#     ticker_names: list[str] | None = None,
#     verbose: bool = True,
# ) -> dict:
#     y = _ensure_2d_y(y)

#     if ticker_names is None:
#         ticker_names = _default_ticker_names(y.shape[1])

#     if len(ticker_names) != y.shape[1]:
#         raise ValueError(
#             f"len(ticker_names)={len(ticker_names)} must match y.shape[1]={y.shape[1]}"
#         )

#     results = {}

#     print("\nPCA + Pearson")
#     r = pca_pearson(X, y, ticker_names=ticker_names)
#     results["pca_pearson"] = r
#     if verbose:
#         for comp in r["results"]:
#             print(
#                 f"  PC{comp['component']:2d} "
#                 f"expvar={comp['explained_var']:.3f}"
#             )
#             for ticker in ticker_names:
#                 vals = comp["by_ticker"][ticker]
#                 sig = "*" if vals["p"] < 0.05 else ""
#                 print(
#                     f"    {ticker:>6s}  r={vals['r']:+.3f}  p={vals['p']:.3f} {sig}"
#                 )

#     print("\nPCA + Spearman")
#     r = spearman_on_pca(X, y, ticker_names=ticker_names)
#     results["pca_spearman"] = r
#     if verbose:
#         for comp in r["results"]:
#             print(
#                 f"  PC{comp['component']:2d} "
#                 f"expvar={comp['explained_var']:.3f}"
#             )
#             for ticker in ticker_names:
#                 vals = comp["by_ticker"][ticker]
#                 sig = "*" if vals["p"] < 0.05 else ""
#                 print(
#                     f"    {ticker:>6s}  rho={vals['r']:+.3f}  p={vals['p']:.3f} {sig}"
#                 )

#     print("\nCCA")
#     r = cca_correlation(X, y, n_components=min(3, y.shape[1]))
#     results["cca"] = r
#     if verbose:
#         for i, cc in enumerate(r["canonical_correlations"], start=1):
#             sig = "*" if cc["p"] < 0.05 else ""
#             print(f"  Canonical {i:2d}  r={cc['r']:+.3f}  p={cc['p']:.3f} {sig}")
#         print(
#             f"  (PCA pre-reduction to {r['pca_components_used']} dims, "
#             f"explaining {r['explained_var_by_pca']:.1%} of variance)"
#         )

#     print("\nRV Coefficient")
#     r = rv_coefficient(X, y)
#     results["rv"] = r
#     if verbose:
#         print(
#             f"  RV={r['rv']:.4f}  p={r['p_value']:.3f} "
#             f"(permutation test, n={r['n_permutations']})"
#         )

#     return results

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _ensure_2d_y(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return y


def _default_ticker_names(n: int) -> list[str]:
    return [f"T{i+1}" for i in range(n)]


def _safe_standardize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    scaler = StandardScaler()
    return scaler.fit_transform(A)


def _bh_fdr(p_values: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev

    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _select_n_pca_by_variance(X: np.ndarray, variance_threshold: float = 0.8, max_components: int | None = None) -> int:
    max_allowed = min(X.shape[0] - 1, X.shape[1])
    if max_components is not None:
        max_allowed = min(max_allowed, max_components)

    pca = PCA(n_components=max_allowed)
    pca.fit(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cumulative, variance_threshold) + 1)
    return max(1, min(n, max_allowed))


# ------------------------------------------------------------
# PCA + correlation
# ------------------------------------------------------------
def pca_correlations(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 10,
    ticker_names: list[str] | None = None,
    method: str = "pearson",
    scale_X: bool = True,
    scale_y: bool = False,
    fdr_correct: bool = True,
) -> dict:
    X = np.asarray(X, dtype=np.float64)
    y = _ensure_2d_y(y)

    if scale_X:
        X = _safe_standardize(X)
    if scale_y:
        y = _safe_standardize(y)

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)

    if ticker_names is None:
        ticker_names = _default_ticker_names(y.shape[1])

    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")

    per_component = []
    all_pvals = []
    pair_index = []

    for i in range(n_components):
        by_ticker = {}
        for j, ticker in enumerate(ticker_names):
            if method == "pearson":
                r, p = stats.pearsonr(Z[:, i], y[:, j])
            else:
                r, p = stats.spearmanr(Z[:, i], y[:, j])
            by_ticker[ticker] = {"r": float(r), "p": float(p)}
            all_pvals.append(float(p))
            pair_index.append((i, ticker))

        per_component.append({
            "component": i + 1,
            "explained_var": float(pca.explained_variance_ratio_[i]),
            "by_ticker": by_ticker,
        })

    if fdr_correct and all_pvals:
        qvals = _bh_fdr(all_pvals)
        for q, (i, ticker) in zip(qvals, pair_index):
            per_component[i]["by_ticker"][ticker]["q"] = float(q)
    else:
        for i, ticker in pair_index:
            per_component[i]["by_ticker"][ticker]["q"] = None

    return {
        "components": n_components,
        "targets": y.shape[1],
        "method": method,
        "scale_X": scale_X,
        "scale_y": scale_y,
        "fdr_correct": fdr_correct,
        "results": per_component,
    }


def pca_pearson(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 10,
    ticker_names: list[str] | None = None,
    scale_X: bool = True,
    scale_y: bool = False,
    fdr_correct: bool = True,
) -> dict:
    return pca_correlations(
        X=X,
        y=y,
        n_components=n_components,
        ticker_names=ticker_names,
        method="pearson",
        scale_X=scale_X,
        scale_y=scale_y,
        fdr_correct=fdr_correct,
    )


def spearman_on_pca(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 10,
    ticker_names: list[str] | None = None,
    scale_X: bool = True,
    scale_y: bool = False,
    fdr_correct: bool = True,
) -> dict:
    return pca_correlations(
        X=X,
        y=y,
        n_components=n_components,
        ticker_names=ticker_names,
        method="spearman",
        scale_X=scale_X,
        scale_y=scale_y,
        fdr_correct=fdr_correct,
    )


# ------------------------------------------------------------
# CCA with stronger safeguards against overfitting
# ------------------------------------------------------------
def cca_correlation(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 1,
    pca_variance_threshold: float = 0.8,
    max_pca_components: int = 10,
    scale_X: bool = True,
    scale_y: bool = True,
) -> dict:
    X = np.asarray(X, dtype=np.float64)
    y = _ensure_2d_y(y)

    if scale_X:
        X = _safe_standardize(X)
    if scale_y:
        y = _safe_standardize(y)

    n_pca = _select_n_pca_by_variance(
        X,
        variance_threshold=pca_variance_threshold,
        max_components=max_pca_components,
    )

    pca = PCA(n_components=n_pca)
    X_reduced = pca.fit_transform(X)

    max_safe_components = min(n_pca, y.shape[1], max(1, X.shape[0] // 5))
    n_components = min(n_components, max_safe_components)

    cca = CCA(n_components=n_components, max_iter=2000)
    cca.fit(X_reduced, y)
    X_c, y_c = cca.transform(X_reduced, y)

    canonical_corrs = []
    for i in range(n_components):
        r, p = stats.pearsonr(X_c[:, i], y_c[:, i])
        canonical_corrs.append({"r": float(r), "p": float(p)})

    return {
        "n_components": n_components,
        "canonical_correlations": canonical_corrs,
        "pca_components_used": int(n_pca),
        "explained_var_by_pca": float(pca.explained_variance_ratio_.sum()),
        "scale_X": scale_X,
        "scale_y": scale_y,
        "pca_variance_threshold": pca_variance_threshold,
        "max_pca_components": max_pca_components,
    }


# ------------------------------------------------------------
# RV coefficient with centering/scaling for more interpretable structure test
# ------------------------------------------------------------
def rv_coefficient(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 999,
    scale_X: bool = True,
    scale_y: bool = True,
    random_state: int = 42,
) -> dict:
    X = np.asarray(X, dtype=np.float64)
    y = _ensure_2d_y(y)

    if scale_X:
        X = _safe_standardize(X)
    if scale_y:
        y = _safe_standardize(y)

    def _rv(A, B):
        SA = A @ A.T
        SB = B @ B.T
        num = np.trace(SA @ SB)
        denom = np.sqrt(np.trace(SA @ SA) * np.trace(SB @ SB))
        return num / denom if denom > 0 else 0.0

    observed_rv = _rv(X, y)

    count_extreme = 0
    rng = np.random.default_rng(random_state)
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        if _rv(X, y_perm) >= observed_rv:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {
        "rv": float(observed_rv),
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
        "scale_X": scale_X,
        "scale_y": scale_y,
    }


# ------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------
def summarize_significant_results(results: dict, alpha: float = 0.05, use_fdr: bool = True) -> list[dict]:
    rows = []
    for comp in results["results"]:
        for ticker, vals in comp["by_ticker"].items():
            crit = vals.get("q") if use_fdr and vals.get("q") is not None else vals["p"]
            if crit < alpha:
                rows.append({
                    "component": comp["component"],
                    "explained_var": comp["explained_var"],
                    "ticker": ticker,
                    "r": vals["r"],
                    "p": vals["p"],
                    "q": vals.get("q"),
                })
    rows.sort(key=lambda d: abs(d["r"]), reverse=True)
    return rows


def run_all(
    X: np.ndarray,
    y: np.ndarray,
    ticker_names: list[str] | None = None,
    verbose: bool = True,
    n_pca_components: int = 10,
    alpha: float = 0.05,
) -> dict:
    y = _ensure_2d_y(y)

    if ticker_names is None:
        ticker_names = _default_ticker_names(y.shape[1])

    if len(ticker_names) != y.shape[1]:
        raise ValueError(
            f"len(ticker_names)={len(ticker_names)} must match y.shape[1]={y.shape[1]}"
        )

    results = {}

    print("\nPCA + Pearson")
    r = pca_pearson(
        X,
        y,
        n_components=n_pca_components,
        ticker_names=ticker_names,
        scale_X=True,
        scale_y=False,
        fdr_correct=True,
    )
    results["pca_pearson"] = r
    if verbose:
        for comp in r["results"]:
            print(f"  PC{comp['component']:2d} expvar={comp['explained_var']:.3f}")
            for ticker in ticker_names:
                vals = comp["by_ticker"][ticker]
                sig = "*" if vals["p"] < alpha else ""
                fdr = f" q={vals['q']:.3f}" if vals.get("q") is not None else ""
                print(f"    {ticker:>6s}  r={vals['r']:+.3f}  p={vals['p']:.3f}{fdr} {sig}")

    print("\nPCA + Spearman")
    r = spearman_on_pca(
        X,
        y,
        n_components=n_pca_components,
        ticker_names=ticker_names,
        scale_X=True,
        scale_y=False,
        fdr_correct=True,
    )
    results["pca_spearman"] = r
    if verbose:
        for comp in r["results"]:
            print(f"  PC{comp['component']:2d} expvar={comp['explained_var']:.3f}")
            for ticker in ticker_names:
                vals = comp["by_ticker"][ticker]
                sig = "*" if vals["p"] < alpha else ""
                fdr = f" q={vals['q']:.3f}" if vals.get("q") is not None else ""
                print(f"    {ticker:>6s}  rho={vals['r']:+.3f}  p={vals['p']:.3f}{fdr} {sig}")

    print("\nCCA")
    r = cca_correlation(
        X,
        y,
        n_components=min(3, y.shape[1]),
        pca_variance_threshold=0.8,
        max_pca_components=10,
        scale_X=True,
        scale_y=True,
    )
    results["cca"] = r
    if verbose:
        for i, cc in enumerate(r["canonical_correlations"], start=1):
            sig = "*" if cc["p"] < alpha else ""
            print(f"  Canonical {i:2d}  r={cc['r']:+.3f}  p={cc['p']:.3f} {sig}")
        print(
            f"  (PCA pre-reduction to {r['pca_components_used']} dims, "
            f"explaining {r['explained_var_by_pca']:.1%} of variance)"
        )

    print("\nRV Coefficient")
    r = rv_coefficient(
        X,
        y,
        n_permutations=999,
        scale_X=True,
        scale_y=True,
        random_state=42,
    )
    results["rv"] = r
    if verbose:
        print(
            f"  RV={r['rv']:.4f}  p={r['p_value']:.3f} "
            f"(permutation test, n={r['n_permutations']})"
        )

    return results