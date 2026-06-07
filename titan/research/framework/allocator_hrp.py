"""Hierarchical Risk Parity (HRP) allocator (audit P3-2).

Lopez de Prado 2016: cluster strategies on the correlation-distance tree,
quasi-diagonalise the covariance so similar streams sit adjacent, then split
capital top-down by recursive bisection (inverse-cluster-variance). Unlike
mean-variance optimisation HRP needs NO matrix inversion, so it is robust to
the ill-conditioning that makes Markowitz weights explode on correlated
strategies -- the failure mode the audit flagged for the all-equity-beta book.

It complements the P3-1 CDaR allocator (which targets tail drawdown directly):
HRP answers "how do I spread risk across a correlated set without trusting a
fragile inverse-covariance?" and reuses the P1-27 Ledoit-Wolf shrunk covariance
(+ optional crisis-correlation overlay) as its conditioning-safe input.

Reference: Lopez de Prado 2016, "Building Diversified Portfolios that
Outperform Out of Sample" (Journal of Portfolio Management).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from titan.research.framework.covariance import shrink_covariance

DEFAULT_LINKAGE = "single"


@dataclass(frozen=True)
class HrpResult:
    """HRP allocation output."""

    weights: dict[str, float]
    cluster_order: list[str]  # quasi-diagonalisation leaf order
    shrinkage: float  # Ledoit-Wolf intensity of the input covariance
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


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 1e-24, None))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _quasi_diag(link: np.ndarray, n_leaves: int) -> list[int]:
    """Leaf order from a linkage matrix (similar items adjacent)."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_ix.max() >= n_leaves:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # make room
        clusters = sort_ix[sort_ix >= n_leaves]
        idx = clusters.index
        rows = clusters.to_numpy() - n_leaves
        sort_ix[idx] = link[rows, 0]  # left child
        right = pd.Series(link[rows, 1], index=idx + 1)  # right child
        sort_ix = pd.concat([sort_ix, right]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _inv_var_weights(cov_sub: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.clip(np.diag(cov_sub), 1e-24, None)
    return ivp / ivp.sum()


def _cluster_var(cov_df: pd.DataFrame, items: list[str]) -> float:
    sub = cov_df.loc[items, items].to_numpy()
    w = _inv_var_weights(sub).reshape(-1, 1)
    return float((w.T @ sub @ w)[0, 0])


def _recursive_bisection(cov_df: pd.DataFrame, order: list[str]) -> pd.Series:
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [
            grp[start:end]
            for grp in clusters
            for start, end in ((0, len(grp) // 2), (len(grp) // 2, len(grp)))
            if len(grp) > 1
        ]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov_df, c0), _cluster_var(cov_df, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w


def _apply_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Iteratively cap weights at ``cap`` and redistribute the excess
    proportionally across the uncapped names (water-filling).
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


def compute_hrp_weights(
    strategy_returns: dict[str, pd.Series],
    *,
    linkage_method: str = DEFAULT_LINKAGE,
    crisis_rho: float | None = None,
    max_weight: float | None = None,
) -> HrpResult:
    """Hierarchical Risk Parity deployment weights (audit P3-2).

    ``crisis_rho`` floors off-diagonal correlations in the input covariance
    (stress the diversification thesis). ``max_weight`` optionally caps single
    -strategy concentration (excess water-filled across the uncapped names).
    """
    df = _aligned_frame(strategy_returns)
    names = list(df.columns)
    if len(names) == 1:
        return HrpResult({names[0]: 1.0}, names, 0.0, int(df.shape[0]))

    sc = shrink_covariance(df, crisis_rho=crisis_rho)
    cov = sc.cov
    cov_df = pd.DataFrame(cov, index=sc.names, columns=sc.names)
    corr = _corr_from_cov(cov)

    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method=linkage_method)
    order = [sc.names[i] for i in _quasi_diag(link, len(sc.names))]

    w = _recursive_bisection(cov_df, order)
    weights = {n: float(w[n]) for n in sc.names}
    if max_weight is not None:
        weights = _apply_cap(weights, max_weight)

    return HrpResult(
        weights=weights,
        cluster_order=order,
        shrinkage=float(sc.shrinkage),
        n_obs=int(sc.n_obs),
    )


__all__ = ["HrpResult", "compute_hrp_weights", "DEFAULT_LINKAGE"]
