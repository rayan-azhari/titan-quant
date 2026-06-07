"""Nested Clustered Optimization (NCO) allocator (audit P3-2).

López de Prado, *Machine Learning for Asset Managers* (2020) ch.7 +
"A Robust Estimator of the Efficient Frontier" (2019). NCO attacks the same
problem the audit flagged for the all-equity-beta book -- Markowitz weights
explode when the covariance is ill-conditioned (correlated streams, small N) --
but goes further than HRP by combining three robustness steps:

  1. DENOISE the correlation matrix with Marčenko-Pastur: the eigenvalues that
     fall inside the random-matrix bulk are pure noise; replacing them with
     their average (constant-residual) strips the sampling noise that makes the
     inverse unstable, while keeping the genuine factor structure.
  2. CLUSTER the denoised correlation into groups of similar streams.
  3. ALLOCATE in two nested stages: optimise WITHIN each cluster on its
     sub-covariance, collapse each cluster to a single synthetic stream, then
     optimise ACROSS the (now small, well-separated) reduced covariance.
     The matrix is only ever inverted on a small, intra-cluster block, so the
     conditioning problem is contained.

The audit (P3-2) explicitly asked for this "pulled forward to small-N" rather
than deferred to 6+ streams -- small-N is exactly where covariance inversion is
least stable, so the denoise + nesting matters most there. It complements the
P3-1 CDaR allocator (tail-drawdown LP) and the HRP allocator (inversion-free
recursive bisection); NCO is the denoise-then-optimise member of the trio and
reuses the P1-27 Ledoit-Wolf shrunk covariance as its conditioning-safe input.

Weights are long-only and sum to 1: an unconstrained min-variance solution that
produces negative weights is clipped to non-negative and renormalised (falling
back to inverse-variance if the block is degenerate), since these are
deployment capital weights, not a long/short overlay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from titan.research.framework.covariance import shrink_covariance

DEFAULT_OBJECTIVE = "min_variance"
_KMEANS_N_INIT = 10
_KMEANS_SEED = 42  # ML reproducibility rule


@dataclass(frozen=True)
class NcoResult:
    """NCO allocation output + diagnostics."""

    weights: dict[str, float]
    clusters: list[list[str]]
    n_clusters: int
    n_factors_kept: int  # MP signal eigenvalues retained (denoise)
    shrinkage: float  # Ledoit-Wolf intensity of the input covariance
    condition_number: float
    objective: str
    n_obs: int


def _aligned_frame(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    names = list(strategy_returns.keys())
    common = None
    for r in strategy_returns.values():
        r = r.dropna()
        common = r.index if common is None else common.intersection(r.index)
    if common is None or len(common) < 30:
        raise ValueError(f"Common index too short: {len(common) if common is not None else 0}")
    return pd.DataFrame({n: strategy_returns[n].reindex(common).fillna(0.0) for n in names})


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 1e-24, None))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _corr_to_cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    return corr * np.outer(std, std)


def marcenko_pastur_denoise(corr: np.ndarray, q: float) -> tuple[np.ndarray, int]:
    """Denoise a correlation matrix via the Marčenko-Pastur constant-residual method.

    Eigenvalues below the MP upper edge ``λ+ = (1 + sqrt(1/q))²`` (with
    ``q = T/N``) lie inside the random-matrix bulk and carry no signal; they
    are replaced by their common average and the matrix is reconstructed +
    rescaled to a unit diagonal. The larger ("signal") eigenvalues are kept.

    Parameters
    ----------
    corr:
        An ``N x N`` correlation matrix (unit diagonal).
    q:
        Observations-per-variable ratio ``T / N`` (> 0). Larger q -> narrower
        noise bulk -> fewer eigenvalues denoised.

    Returns:
    -------
    (corr_denoised, n_factors_kept)
        The denoised correlation matrix and the number of signal eigenvalues
        retained (always >= 1).
    """
    if q <= 0:
        raise ValueError(f"q (T/N) must be > 0, got {q}")
    n = corr.shape[0]
    if n < 2:
        return corr.copy(), n

    evals, evecs = np.linalg.eigh(corr)  # ascending
    lambda_plus = (1.0 + np.sqrt(1.0 / q)) ** 2
    n_factors = int(np.sum(evals > lambda_plus))
    n_factors = max(1, min(n_factors, n))  # keep >= 1 signal eigenvalue

    if n_factors >= n:
        return corr.copy(), n_factors  # nothing inside the bulk

    evals_denoised = evals.copy()
    # The smallest (n - n_factors) eigenvalues are the noise bulk -> average.
    n_noise = n - n_factors
    evals_denoised[:n_noise] = float(evals[:n_noise].mean())

    corr1 = (evecs * evals_denoised) @ evecs.T
    corr1 = _cov_to_corr(corr1)  # rescale to unit diagonal
    return corr1, n_factors


def _inv_var_weights(cov_sub: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.clip(np.diag(cov_sub), 1e-24, None)
    return ivp / ivp.sum()


def _min_var_weights(cov_sub: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance weights with robust fallbacks.

    Unconstrained min-variance ``w = Σ⁻¹1 / 1ᵀΣ⁻¹1`` can be negative; for
    deployment capital weights we clip to non-negative and renormalise, and
    fall back to inverse-variance if the block is singular or degenerates.
    """
    n = cov_sub.shape[0]
    if n == 1:
        return np.array([1.0])
    try:
        inv = np.linalg.inv(cov_sub)
    except np.linalg.LinAlgError:
        return _inv_var_weights(cov_sub)
    ones = np.ones(n)
    w = inv @ ones
    s = w.sum()
    if not np.isfinite(s) or abs(s) < 1e-18:
        return _inv_var_weights(cov_sub)
    w = w / s
    w = np.clip(w, 0.0, None)  # long-only
    tot = w.sum()
    if tot <= 1e-18:
        return _inv_var_weights(cov_sub)
    return w / tot


def _objective_weights(cov_sub: np.ndarray, objective: str) -> np.ndarray:
    if objective == "min_variance":
        return _min_var_weights(cov_sub)
    if objective == "inverse_variance":
        return _inv_var_weights(cov_sub)
    raise ValueError(f"objective must be 'min_variance' or 'inverse_variance', got {objective!r}")


def _cluster_corr(
    corr: np.ndarray, names: list[str], *, max_clusters: int | None
) -> list[list[str]]:
    """Cluster the correlation matrix into groups of similar streams.

    Uses KMeans on the correlation-distance observations (LdP's clusterKMeansBase),
    selecting the number of clusters K in ``[2, max_clusters]`` that maximises the
    silhouette t-stat (mean / std of silhouette samples). For N <= 2 returns a
    single cluster (NCO degenerates to a direct optimisation).
    """
    n = len(names)
    if n <= 2:
        return [list(names)]

    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    upper = n - 1
    k_max = upper if max_clusters is None else min(max_clusters, upper)
    k_max = max(2, k_max)

    best_labels: np.ndarray | None = None
    best_score = -np.inf
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=_KMEANS_N_INIT, random_state=_KMEANS_SEED).fit(dist)
        labels = km.labels_
        if len(set(labels)) < 2:
            continue
        silh = silhouette_samples(dist, labels)
        std = silh.std()
        score = (silh.mean() / std) if std > 1e-12 else silh.mean()
        if score > best_score:
            best_score = score
            best_labels = labels

    if best_labels is None:
        return [list(names)]

    clusters: dict[int, list[str]] = {}
    for name, lbl in zip(names, best_labels):
        clusters.setdefault(int(lbl), []).append(name)
    return [clusters[k] for k in sorted(clusters)]


def _apply_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Iteratively cap weights at ``cap`` and water-fill the excess across the
    uncapped names (same scheme as the HRP allocator).
    """
    w = dict(weights)
    for _ in range(100):
        over = {k: v for k, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for k in over:
            w[k] = cap
        free = {k: w[k] for k in w if w[k] < cap - 1e-12}
        free_total = sum(free.values())
        if not free:
            break
        if free_total <= 0:
            for k in free:
                w[k] += excess / len(free)
        else:
            for k in free:
                w[k] += excess * w[k] / free_total
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def compute_nco_weights(
    strategy_returns: dict[str, pd.Series],
    *,
    objective: str = DEFAULT_OBJECTIVE,
    crisis_rho: float | None = None,
    max_weight: float | None = None,
    max_clusters: int | None = None,
    denoise: bool = True,
) -> NcoResult:
    """Nested Clustered Optimization deployment weights (audit P3-2).

    Parameters
    ----------
    strategy_returns:
        Mapping ``name -> per-bar return Series``. Aligned on the common index.
    objective:
        ``"min_variance"`` (default) or ``"inverse_variance"`` for both the
        intra- and inter-cluster stages.
    crisis_rho:
        Floor off-diagonal correlations at this level in the input covariance
        (stress the diversification thesis), as in the HRP/ERC allocators.
    max_weight:
        Optional single-strategy cap, water-filled across the uncapped names.
    max_clusters:
        Cap on the number of clusters K considered (default ``N - 1``).
    denoise:
        Apply Marčenko-Pastur denoising before clustering/optimising
        (default True). Set False to optimise on the raw shrunk covariance.

    Returns:
    -------
    NcoResult with weights, the discovered clusters, and diagnostics.
    """
    df = _aligned_frame(strategy_returns)
    names = list(df.columns)
    if len(names) == 1:
        return NcoResult({names[0]: 1.0}, [names], 1, 1, 0.0, 1.0, objective, int(df.shape[0]))

    sc = shrink_covariance(df, crisis_rho=crisis_rho)
    names = list(sc.names)
    cov = sc.cov
    std = np.sqrt(np.clip(np.diag(cov), 1e-24, None))
    corr = _cov_to_corr(cov)

    n_factors = corr.shape[0]
    if denoise:
        q = float(sc.n_obs) / float(len(names))
        corr, n_factors = marcenko_pastur_denoise(corr, q)
        cov = _corr_to_cov(corr, std)

    clusters = _cluster_corr(corr, names, max_clusters=max_clusters)
    pos = {name: i for i, name in enumerate(names)}

    # Intra-cluster weights -> build the (N x K) loading matrix.
    k = len(clusters)
    loadings = np.zeros((len(names), k))
    for ci, members in enumerate(clusters):
        idx = [pos[m] for m in members]
        sub = cov[np.ix_(idx, idx)]
        w_intra = _objective_weights(sub, objective)
        for j, m in enumerate(members):
            loadings[pos[m], ci] = w_intra[j]

    # Reduced (cluster-level) covariance, then inter-cluster weights.
    reduced_cov = loadings.T @ cov @ loadings
    w_inter = _objective_weights(reduced_cov, objective)

    final = loadings @ w_inter
    total = final.sum()
    if total <= 1e-18:
        final = np.full(len(names), 1.0 / len(names))
        total = 1.0
    final = final / total

    weights = {name: float(final[pos[name]]) for name in names}
    if max_weight is not None:
        weights = _apply_cap(weights, max_weight)

    return NcoResult(
        weights=weights,
        clusters=clusters,
        n_clusters=k,
        n_factors_kept=int(n_factors),
        shrinkage=float(sc.shrinkage),
        condition_number=float(sc.condition_number),
        objective=objective,
        n_obs=int(sc.n_obs),
    )


__all__ = [
    "NcoResult",
    "compute_nco_weights",
    "marcenko_pastur_denoise",
    "DEFAULT_OBJECTIVE",
]
