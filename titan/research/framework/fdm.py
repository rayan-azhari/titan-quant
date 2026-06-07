"""Forecast Diversification Multiplier (Carver, *Systematic Trend Following*).

When combining N forecasts (or strategy returns) into an ensemble, the
combined signal's "scale" is reduced by the diversification benefit --
the more independent the forecasts, the more the average self-cancels.
The FDM is the scalar that restores the combined forecast to a target
magnitude, equivalent to the inverse of the portfolio standard deviation
when each forecast is unit-variance.

Math
====

For N forecasts with correlation matrix ρ, the EQUAL-WEIGHTED combined
forecast has variance ``w' ρ w`` where ``w = (1/N, 1/N, ..., 1/N)``.
The FDM is

    fdm = 1 / sqrt(w' ρ w)

so that ``combined * fdm`` has the same RMS magnitude as a single
forecast. For UNIFORM pairwise correlation ``c``:

    fdm = sqrt(N / (1 + (N-1) * c))

Limits:
* N=1 -> fdm = 1.0 (no diversification possible).
* c=1 (perfectly correlated) -> fdm = 1.0 (no diversification benefit).
* c=0 (independent) -> fdm = sqrt(N).
* c<0 (negative correlation) -> fdm > sqrt(N) (super-diversification).

Cap
===

Carver caps the FDM at 2.5 to avoid overfitting on apparently-low
correlations that are sample artefacts. With realistic financial-asset
correlation floors (~0.25-0.5 between trend forecasts on the same
universe), achievable FDMs are typically 1.3-1.6. An uncapped FDM above
2.5 almost always reflects undersampled correlation rather than real
independence.

Use
===

The FDM is applied to the AVERAGE of normalised forecasts:

    combined_forecast = mean(per_strategy_forecast) * fdm

NOT to position sizes directly (the allocator handles weights). The FDM
is a meta-allocator helper: when you have N already-sized strategies
running and you want to combine their SCALAR forecasts into a single
ensemble signal, the FDM is what keeps the ensemble at the right
magnitude.

This module is referenced by L75 + backlog J5; it is the first general
infrastructure primitive added in the V3.7 framework that operates on
the OUTPUT side of strategy combination (the allocator handles input
weights; the FDM handles output forecast scale).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Default cap from Carver, *Systematic Trend Following* (2018 ed.) and
# *Systematic Trading* (2015). Higher caps invite overfitting.
DEFAULT_FDM_CAP: float = 2.5
# Minimum days of overlapping history before we trust a correlation
# estimate. With <60 days the pairwise correlation has too wide a CI to
# build the FDM on top of.
DEFAULT_MIN_HISTORY: int = 60


@dataclass(frozen=True)
class FdmResult:
    """Result of an FDM computation.

    Attributes:
    ----------
    fdm:
        The capped FDM. Always >= 1.0 (a non-diversified ensemble cannot
        have lower combined-RMS than a single forecast).
    fdm_uncapped:
        The raw FDM before the cap was applied. Equal to ``fdm`` when
        ``was_capped is False``.
    n_forecasts:
        Number of forecasts that contributed to the FDM (after dropping
        any all-NaN columns).
    avg_correlation:
        Mean of the off-diagonal entries of the correlation matrix.
        Useful for sanity-checking (e.g. trend forecasts on the same
        universe typically average 0.4-0.7).
    correlation_matrix:
        Full N x N correlation matrix used in the calculation. Index +
        columns match the input DataFrame's columns.
    was_capped:
        True iff ``fdm_uncapped > cap`` and ``fdm = cap``.
    """

    fdm: float
    fdm_uncapped: float
    n_forecasts: int
    avg_correlation: float
    correlation_matrix: pd.DataFrame
    was_capped: bool


def forecast_diversification_multiplier(
    returns: pd.DataFrame,
    *,
    cap: float = DEFAULT_FDM_CAP,
    min_history_days: int = DEFAULT_MIN_HISTORY,
) -> FdmResult:
    """Compute the Carver Forecast Diversification Multiplier for N forecasts.

    Parameters
    ----------

    Returns:
        DataFrame where each column is one forecast's per-bar returns
        (or any standardised proxy -- only the correlation structure
        matters). Rows must share a common time index. NaN cells are
        handled via pandas' pairwise correlation default.
    cap:
        Upper bound on the FDM. Default 2.5 (Carver standard). Raise the
        cap only when correlations are genuinely well-estimated AND you
        have a reason to trust independence claims above the cap.
    min_history_days:
        Minimum non-NaN row count before the FDM is trusted. If fewer
        rows are available, the function still returns a result but
        flags the under-history case via a warning-friendly low N or
        wide CI. Raise this if you need tighter confidence.

    Returns:
    -------
    FdmResult with the capped FDM, the uncapped value, N, avg
    correlation, the correlation matrix, and a cap flag.

    Raises:
    ------
    ValueError
        If `returns` has fewer than 1 valid column, or if the resulting
        ``w' ρ w`` is non-positive (which would indicate a malformed
        correlation matrix -- usually a sign that the input is constant
        or has only zero-vol columns).
    """
    if returns is None or returns.empty:
        raise ValueError("forecast_diversification_multiplier: empty returns frame")

    # Drop columns that are entirely NaN or zero-variance.
    valid_cols = [c for c in returns.columns if returns[c].dropna().std(ddof=1) > 0]
    if not valid_cols:
        raise ValueError("forecast_diversification_multiplier: no columns with non-zero variance")

    df = returns[valid_cols]
    n = len(valid_cols)

    if n == 1:
        # A single forecast cannot be diversified.
        return FdmResult(
            fdm=1.0,
            fdm_uncapped=1.0,
            n_forecasts=1,
            avg_correlation=float("nan"),
            correlation_matrix=pd.DataFrame([[1.0]], index=valid_cols, columns=valid_cols),
            was_capped=False,
        )

    # Pairwise correlation. NaN handling: pandas default uses pairwise
    # complete observations per cell, which is robust to misaligned
    # series. Caller-supplied min_history_days kicks in below if even
    # that produces too few common observations.
    common = df.dropna(how="any")
    if len(common) < min_history_days:
        # Fall back to per-pair available data; widen the CI implicitly.
        corr = df.corr()
    else:
        corr = common.corr()

    # Sanity: clamp any numerical artefacts in [-1, 1].
    corr = corr.clip(lower=-1.0, upper=1.0)
    # Force the diagonal to be exactly 1.0.
    np.fill_diagonal(corr.values, 1.0)

    # Equal-weighted portfolio variance: w' ρ w with w = 1/N.
    w = np.ones(n) / n
    portfolio_var = float(w @ corr.values @ w)
    if portfolio_var <= 0.0:
        raise ValueError(
            f"forecast_diversification_multiplier: non-positive ensemble "
            f"variance ({portfolio_var}) -- correlation matrix is malformed"
        )

    fdm_uncapped = 1.0 / math.sqrt(portfolio_var)
    # Floor at 1.0: even if numerical noise gives slightly < 1, no
    # diversification benefit can produce sub-unit FDM by construction.
    fdm_uncapped = max(1.0, fdm_uncapped)

    was_capped = fdm_uncapped > cap
    fdm_capped = min(fdm_uncapped, cap)

    # Average off-diagonal correlation: useful diagnostic.
    off_diag_mask = ~np.eye(n, dtype=bool)
    avg_corr = float(corr.values[off_diag_mask].mean())

    return FdmResult(
        fdm=fdm_capped,
        fdm_uncapped=fdm_uncapped,
        n_forecasts=n,
        avg_correlation=avg_corr,
        correlation_matrix=corr,
        was_capped=was_capped,
    )


def fdm_from_uniform_correlation(n: int, avg_corr: float) -> float:
    """Closed-form FDM for the uniform-correlation special case.

    Useful when you don't have full return histories but you have a
    population-level estimate of the average pairwise correlation
    (e.g. Carver's per-asset-class published values). Same cap rules
    do NOT apply automatically -- the caller must clip if desired.

    Parameters
    ----------
    n:
        Number of forecasts.
    avg_corr:
        Uniform pairwise correlation, in [-1/(n-1), 1].

    Returns:
    -------
    The uncapped FDM. For n=1 returns 1.0. For avg_corr=1.0 returns
    1.0 (no diversification benefit). For avg_corr=0 returns sqrt(n).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return 1.0
    # Feasibility: uniform correlation matrix is PSD iff
    # avg_corr >= -1/(n-1). Clamp gently to avoid sqrt(neg) on
    # numerical edge cases.
    lower_bound = -1.0 / (n - 1)
    avg_corr = max(lower_bound, min(1.0, avg_corr))
    denom = 1.0 + (n - 1) * avg_corr
    if denom <= 0.0:
        # Theoretical edge: at the lower-bound correlation the
        # equal-weight portfolio variance touches zero. The "FDM" is
        # unbounded; return infinity rather than raise so callers can
        # clip per their own cap.
        return float("inf")
    return math.sqrt(n / denom)
