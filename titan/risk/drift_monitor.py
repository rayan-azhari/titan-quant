"""Live-vs-predicted drawdown drift monitor (audit P0-9).

The mandate (RoR≈0 / MaxDD<20%) is only meaningful if we MEASURE whether the
live book is drawing down worse than the MC predicted. This module is the pure
core of that measurement:

  * ``predicted_maxdd_band`` -- block-bootstrap the portfolio's historical
    returns over the deployment horizon and return the p95 / p99 MaxDD band
    (the tail the live realised MaxDD is checked against).
  * ``realised_rolling_maxdd`` -- the MaxDD of the live NAV over a trailing
    window.
  * ``drift_band_decision`` -- the alert / auto-de-risk decision with
    hysteresis: ALERT when realised MaxDD breaches the p95 band; AUTO-DE-RISK
    (halve the PRM scale) when it breaches the p99 band; hold the de-risk until
    realised MaxDD recovers back inside p95.

All MaxDD values are non-positive fractions (e.g. -0.18). A realised MaxDD is a
"breach" when it is MORE NEGATIVE than the band value. Everything here is pure
(seeded MC, no I/O / globals) so the policy is exhaustively unit-tested. The
heavy band MC is run periodically by ``scripts/monitor_live_drift.py``; the
cheap realised-MaxDD comparison + the de-risk fold live in the PRM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Defaults: 10y daily horizon (matches GATE_V38), 20-bar blocks (≈1 month).
DEFAULT_HORIZON_BARS = 2520
DEFAULT_BLOCK_SIZE = 20
DEFAULT_N_PATHS = 2000
DEFAULT_DERISK_SCALE = 0.5


def realised_rolling_maxdd(nav: pd.Series, window_bars: int) -> float:
    """MaxDD (non-positive) of the live NAV over the trailing ``window_bars``."""
    s = nav.dropna()
    if window_bars > 0:
        s = s.iloc[-window_bars:]
    if len(s) < 2:
        return 0.0
    eq = s.to_numpy(dtype=float)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def predicted_maxdd_band(
    returns,
    *,
    underlying_returns=None,
    portfolio_weights=None,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_paths: int = DEFAULT_N_PATHS,
    seed: int = 42,
    percentiles: tuple[int, ...] = (95, 99),
) -> dict[int, float]:
    """Block-bootstrap over the horizon; return the MaxDD band.

    Two operating modes, selected by whether ``underlying_returns`` is supplied:

    **Audit-A6 mode** (``underlying_returns`` + ``portfolio_weights`` provided):
        Bootstrap rows from the per-strategy return matrix with SHARED block
        indices, then combine with ``portfolio_weights`` to get synthetic
        portfolio returns per path. Shared draws preserve the cross-strategy
        correlation structure (a bad regime hits all strategies at once), so
        the MC captures correlation spikes that bootstrapping the combined
        portfolio series cannot. This is the recommended path for any live
        multi-strategy book.

    **Legacy mode** (only ``returns`` provided):
        Bootstrap the combined portfolio returns directly. This understates tail
        risk because the correlation structure is already baked into the series
        and bad-regime draws are diluted. Kept for backward compatibility.

    Parameters
    ----------

    Returns:
        1-D array of combined portfolio returns (used in legacy mode; ignored
        when ``underlying_returns`` is supplied).
    underlying_returns:
        (T × N) array-like of per-strategy daily returns, one column per
        strategy. Rows must be aligned (same dates).
    portfolio_weights:
        Length-N array of portfolio weights, one per column of
        ``underlying_returns``. Need not sum to 1 (normalised internally).
    """
    if underlying_returns is not None:
        # Audit-A6 path: shared-block bootstrap over the strategy return matrix.
        U = np.asarray(underlying_returns, dtype=float)
        w = np.asarray(portfolio_weights, dtype=float)
        w = w / w.sum()
        # Drop rows where ANY strategy has a non-finite return.
        finite_mask = np.isfinite(U).all(axis=1)
        U = U[finite_mask]
        n_obs, _n_strat = U.shape
        if n_obs < block_size + 1 or horizon_bars < 2:
            return {p: 0.0 for p in percentiles}
        rng = np.random.default_rng(seed)
        n_blocks = (horizon_bars + block_size - 1) // block_size
        n_available = n_obs - block_size + 1
        maxdds = np.empty(n_paths, dtype=float)
        for i in range(n_paths):
            # Shared block indices preserve cross-strategy correlation.
            starts = rng.integers(0, n_available, size=n_blocks)
            path_mat = np.concatenate([U[s : s + block_size] for s in starts])[
                :horizon_bars
            ]  # (horizon_bars, N)
            path_port = path_mat @ w  # (horizon_bars,) weighted portfolio return
            eq = np.cumprod(1.0 + path_port)
            peak = np.maximum.accumulate(eq)
            maxdds[i] = float((eq / peak - 1.0).min())
        return {p: float(np.percentile(maxdds, 100 - p)) for p in percentiles}

    # Legacy path: bootstrap combined portfolio returns directly.
    # Understates tail risk (see docstring). Use underlying_returns when possible.
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < block_size + 1 or horizon_bars < 2:
        return {p: 0.0 for p in percentiles}
    rng = np.random.default_rng(seed)
    n_blocks = (horizon_bars + block_size - 1) // block_size
    n_available = len(r) - block_size + 1
    maxdds = np.empty(n_paths, dtype=float)
    for i in range(n_paths):
        starts = rng.integers(0, n_available, size=n_blocks)
        path = np.concatenate([r[s : s + block_size] for s in starts])[:horizon_bars]
        eq = np.cumprod(1.0 + path)
        peak = np.maximum.accumulate(eq)
        maxdds[i] = float((eq / peak - 1.0).min())
    # Severity percentile p -> the (100-p)th percentile of the (negative) array:
    # band[95] = 5th percentile (only 5% of paths deeper).
    return {p: float(np.percentile(maxdds, 100 - p)) for p in percentiles}


@dataclass(frozen=True)
class DriftDecision:
    """Outcome of comparing realised MaxDD to the predicted band."""

    alert: bool
    derisk: bool
    drift_scale: float
    reason: str


def drift_band_decision(
    realised_maxdd: float,
    band_p95: float,
    band_p99: float,
    *,
    currently_derisked: bool,
    derisk_scale: float = DEFAULT_DERISK_SCALE,
) -> DriftDecision:
    """Alert / auto-de-risk policy (P0-9), with hysteresis.

    All inputs are non-positive MaxDD fractions; a realised MaxDD "breaches" a
    band when it is more negative than the band value.

      * realised <= p99 band            -> ALERT + DE-RISK (scale = derisk_scale)
      * already de-risked & realised
        still <= p95 band               -> HOLD the de-risk (hysteresis)
      * realised <= p95 band            -> ALERT only (no de-risk)
      * otherwise                       -> normal (scale 1.0)

    De-risk releases only once realised MaxDD recovers back ABOVE the p95 band,
    so a book hovering near p99 doesn't flap on/off.
    """
    if realised_maxdd <= band_p99:
        return DriftDecision(
            True,
            True,
            derisk_scale,
            f"realised MaxDD {realised_maxdd:+.2%} <= p99 band {band_p99:+.2%} -- auto-de-risk",
        )
    if currently_derisked and realised_maxdd <= band_p95:
        return DriftDecision(
            True,
            True,
            derisk_scale,
            f"holding de-risk: realised {realised_maxdd:+.2%} still <= p95 band {band_p95:+.2%}",
        )
    if realised_maxdd <= band_p95:
        return DriftDecision(
            True,
            False,
            1.0,
            f"realised MaxDD {realised_maxdd:+.2%} <= p95 band {band_p95:+.2%} -- alert",
        )
    return DriftDecision(False, False, 1.0, "within predicted band")
