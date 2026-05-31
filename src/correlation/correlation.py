import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def pca_pearson(X: np.ndarray, y: np.ndarray, n_components: int = 10) -> dict:
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)  # (N, n_components)

    pearson_r = []
    p_values = []
    for i in range(n_components):
        r, p = stats.pearsonr(Z[:, i], y)
        pearson_r.append(r)
        p_values.append(p)

    return {
        "components": n_components,
        "pearson_r": pearson_r,
        "p_values": p_values,
        "explained_var": pca.explained_variance_ratio_.tolist(),
    }


def spearman_on_pca(X: np.ndarray, y: np.ndarray, n_components: int = 10) -> dict:
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)

    spearman_r = []
    p_values = []
    for i in range(n_components):
        r, p = stats.spearmanr(Z[:, i], y)
        spearman_r.append(r)
        p_values.append(p)

    return {
        "components": n_components,
        "spearman_r": spearman_r,
        "p_values": p_values,
        "explained_var": pca.explained_variance_ratio_.tolist(),
    }


def cca_correlation( X: np.ndarray, y: np.ndarray, n_components: int = 1, ) -> dict:
    y_col = y.reshape(-1, 1)

    n_pca = min(50, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_pca)
    X_reduced = pca.fit_transform(X)

    n_components = min(n_components, n_pca)
    cca = CCA(n_components=n_components)
    cca.fit(X_reduced, y_col)

    X_c, y_c = cca.transform(X_reduced, y_col)

    canonical_corrs = []
    for i in range(n_components):
        r, p = stats.pearsonr(X_c[:, i], y_c[:, i])
        canonical_corrs.append({"r": r, "p": p})

    return {
        "n_components": n_components,
        "canonical_correlations": canonical_corrs,
        "pca_components_used": n_pca,
        "explained_var_by_pca": pca.explained_variance_ratio_.sum(),
    }


def rv_coefficient(X: np.ndarray, y: np.ndarray) -> dict:
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    def _rv(A, B):
        SA = A @ A.T
        SB = B @ B.T
        num = np.trace(SA @ SB)
        denom = np.sqrt(np.trace(SA @ SA) * np.trace(SB @ SB))
        return num / denom if denom > 0 else 0.0

    observed_rv = _rv(X, y)

    n_permutations = 999
    count_extreme = 0
    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        if _rv(X, y_perm) >= observed_rv:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {"rv": observed_rv, "p_value": p_value, "n_permutations": n_permutations}


def run_all(X: np.ndarray, y: np.ndarray, verbose: bool = True) -> dict:
    results = {}

    print("\nPCA + Pearson")
    r = pca_pearson(X, y)
    results["pca_pearson"] = r
    if verbose:
        for i in range(r["components"]):
            sig = "*" if r["p_values"][i] < 0.05 else ""
            print(f"  PC{i+1:2d}  r={r['pearson_r'][i]:+.3f}  p={r['p_values'][i]:.3f} expvar={r['explained_var'][i]:.3f}  {sig}")

    print("\nPCA + Spearman")
    r = spearman_on_pca(X, y)
    results["pca_spearman"] = r
    if verbose:
        for i in range(r["components"]):
            sig = "*" if r["p_values"][i] < 0.05 else ""
            print( f"  PC{i+1:2d}  ρ={r['spearman_r'][i]:+.3f}  p={r['p_values'][i]:.3f}  {sig}" )

    print("\nCCA")
    r = cca_correlation(X, y, n_components=1)
    results["cca"] = r
    if verbose:
        cc = r["canonical_correlations"][0]
        print(f"  Canonical correlation r={cc['r']:+.3f}  p={cc['p']:.3f}")
        print(f"  (PCA pre-reduction to {r['pca_components_used']} dims, explaining {r['explained_var_by_pca']:.1%} of variance)")

    print("\nRV Coefficient")
    r = rv_coefficient(X, y)
    results["rv"] = r
    if verbose:
        print(f"  RV={r['rv']:.4f}  p={r['p_value']:.3f} (permutation test, n={r['n_permutations']})")

    return results