"""Shrinkage covariance estimation for the small-N portfolio (audit P1-27).

At N~3 live strategies with a 60-obs window, the raw sample covariance is noisy
and often ill-conditioned -- ERC / risk-contribution math built on it inverts an
unstable matrix and over-fits calm-period correlations. This module provides a
robust estimator used by both the ERC allocator and the PRM's
``_risk_contributions``:

  * Ledoit-Wolf shrinkage of the sample covariance toward a structured target
    (sklearn ``LedoitWolf``) -- the closed-form optimal intensity for small N.
  * An optional CRISIS-correlation overlay: floor the off-diagonal correlations
    at ``crisis_rho`` (e.g. 0.8) so the estimate doesn't assume calm-period
    diversification survives a crash (correlations -> 1 exactly when you need
    the diversification).
  * A PSD / condition guard: symmetrise + eigenvalue-clip to the nearest PSD
    matrix and flag ill-conditioning so the caller can fall back to a
    diagonal (inverse-vol) allocation rather than trust a degenerate inverse.

Reference: Ledoit & Wolf 2004, "Honey, I Shrunk the Sample Covariance Matrix."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

DEFAULT_CONDITION_CAP = 1e6
DEFAULT_CRISIS_RHO = 0.8


@dataclass(frozen=True)
class ShrunkCovariance:
    """Shrunk covariance + conditioning diagnostics."""

    cov: np.ndarray
    names: list[str]
    shrinkage: float  # Ledoit-Wolf intensity in [0, 1]
    condition_number: float
    is_well_conditioned: bool
    n_obs: int


def _nearest_psd(cov: np.ndarray) -> np.ndarray:
    """Symmetrise + clip eigenvalues to a small positive floor -> nearest PSD."""
    c = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(c)
    floor = max(1e-16, float(vals.max()) * 1e-12) if vals.size else 1e-16
    vals = np.clip(vals, floor, None)
    return (vecs * vals) @ vecs.T


def _apply_crisis_overlay(cov: np.ndarray, crisis_rho: float) -> np.ndarray:
    """Floor off-diagonal correlations at ``crisis_rho`` (vols unchanged)."""
    d = np.sqrt(np.clip(np.diag(cov), 1e-24, None))
    corr = cov / np.outer(d, d)
    n = corr.shape[0]
    off = ~np.eye(n, dtype=bool)
    corr[off] = np.maximum(corr[off], crisis_rho)
    np.fill_diagonal(corr, 1.0)
    return corr * np.outer(d, d)


def shrink_covariance(
    returns: pd.DataFrame,
    *,
    crisis_rho: float | None = None,
    condition_cap: float = DEFAULT_CONDITION_CAP,
) -> ShrunkCovariance:
    """Ledoit-Wolf shrunk covariance of ``returns`` (columns = strategies),
    with an optional crisis-correlation overlay + PSD/condition guard.
    """
    df = returns.dropna(how="any")
    names = list(df.columns)
    x = df.to_numpy(dtype=float)
    n_obs = x.shape[0]
    if x.shape[1] == 1:
        cov = np.array([[float(np.var(x[:, 0], ddof=1))]])
        shrinkage = 0.0
    else:
        lw = LedoitWolf().fit(x)
        cov = np.asarray(lw.covariance_, dtype=float).copy()
        shrinkage = float(lw.shrinkage_)
    if crisis_rho is not None and len(names) > 1:
        cov = _apply_crisis_overlay(cov, crisis_rho)
    cov = _nearest_psd(cov)
    vals = np.linalg.eigvalsh(cov)
    mn, mx = float(vals.min()), float(vals.max())
    condition_number = (mx / mn) if mn > 0 else float("inf")
    is_well_conditioned = mn > 0 and condition_number < condition_cap
    return ShrunkCovariance(
        cov=cov,
        names=names,
        shrinkage=shrinkage,
        condition_number=condition_number,
        is_well_conditioned=is_well_conditioned,
        n_obs=n_obs,
    )


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """Diagonal (inverse-vol) weights -- the fallback when ``cov`` is
    ill-conditioned (ignores the unstable off-diagonal correlations).
    """
    vol = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    inv = np.where(vol > 1e-18, 1.0 / vol, 0.0)
    total = inv.sum()
    if total <= 0:
        return np.full(len(vol), 1.0 / len(vol))
    return inv / total


__all__ = [
    "ShrunkCovariance",
    "DEFAULT_CRISIS_RHO",
    "shrink_covariance",
    "inverse_vol_weights",
]
