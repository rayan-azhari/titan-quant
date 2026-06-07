"""Deflated Sharpe Ratio (Bailey & López de Prado 2014) -- standardised.

Specified in directives/Methodology Audit & Unified Framework 2026-05-14.md
§2.5. Fixes gaps E1 (normal-returns assumption), E2 (sr_var source),
E3 (N count ambiguity).

The DSR adjusts a strategy's observed Sharpe for the multi-testing bias
introduced by selecting the best of N parameter cells (or N screened
instruments). Key inputs:

    sr_hat              observed annualised Sharpe of THIS trial
    sr_var_across_trials variance of Sharpes across the FULL sweep (not survivors only)
    skew, kurt          skew + Pearson kurt of THIS trial's return series (NOT survivors)
    T                   number of observations in THIS trial's return series
    N                   total number of trials in the sweep

Returns a probability in [0, 1]. Deployment gate: dsr_prob >= 0.95.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329
E_CONST = math.e


@dataclass(frozen=True)
class DsrResult:
    """Single-cell DSR output. The struct is what the audit emits."""

    sharpe: float
    n_trials: int
    sr_var_across_trials: float
    e_max_sr: float  # expected null max SR over N trials
    skew: float
    kurt: float  # Pearson kurt (3 = normal)
    n_obs: int
    z: float  # variance-stabilised gap
    dsr_prob: float  # probability the true Sharpe > 0 after deflation
    survivors_only: bool  # True if sr_var was computed from survivors (optimistic)


def _safe_skew_kurt(returns: pd.Series | np.ndarray) -> tuple[float, float]:
    """Compute sample skew + Pearson kurt with guards for degenerate input."""
    arr = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(arr) < 30:
        return 0.0, 3.0
    sd = arr.std(ddof=1)
    if sd < 1e-12:
        return 0.0, 3.0
    skew = float(stats.skew(arr, bias=False))
    # Pearson kurt (3 for normal). scipy's default is excess kurt (Fisher),
    # so add 3.
    kurt_excess = float(stats.kurtosis(arr, fisher=True, bias=False))
    kurt = 3.0 + kurt_excess
    return skew, kurt


def deflated_sharpe(
    sr_hat: float,
    *,
    sr_var_across_trials: float,
    returns: pd.Series | np.ndarray,
    n_trials: int,
    survivors_only: bool = False,
) -> DsrResult:
    """Bailey & López de Prado 2014 DSR with skew + kurt from the actual
    return distribution (no normal-returns assumption).

    Parameters
    ----------
    sr_hat:
        Observed annualised Sharpe.
    sr_var_across_trials:
        Variance of Sharpes across the FULL sweep. If only survivors are
        available, pass `survivors_only=True` -- the result is flagged
        as OPTIMISTIC (true variance is larger, true DSR is worse).

    Returns:
        This trial's per-bar (or per-day) return series. Skew + kurt are
        estimated from this.
    n_trials:
        Total cells in the sweep (NOT survivors).

    Returns:
    -------
    DsrResult with the full diagnostic breakdown.
    """
    if n_trials < 2 or sr_var_across_trials <= 0:
        return DsrResult(
            sharpe=sr_hat,
            n_trials=n_trials,
            sr_var_across_trials=sr_var_across_trials,
            e_max_sr=0.0,
            skew=0.0,
            kurt=3.0,
            n_obs=len(pd.Series(returns).dropna()),
            z=0.0,
            dsr_prob=0.0,
            survivors_only=survivors_only,
        )

    sr_std = float(math.sqrt(sr_var_across_trials))
    e_max_sr = sr_std * (
        (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (n_trials * E_CONST))
    )
    skew, kurt = _safe_skew_kurt(returns)
    n_obs = len(pd.Series(returns).dropna())
    T = max(n_obs, 30)
    # Bailey & López de Prado 2014 eq. (4). Variance-stabilised gap z is:
    #   z = (sr_hat - e_max_sr) / sqrt( (1 - skew*sr_hat + (kurt-1)/4 * sr_hat^2) / (T-1) )
    denom_inside = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if denom_inside <= 0 or T < 2:
        return DsrResult(
            sharpe=sr_hat,
            n_trials=n_trials,
            sr_var_across_trials=sr_var_across_trials,
            e_max_sr=float(e_max_sr),
            skew=skew,
            kurt=kurt,
            n_obs=n_obs,
            z=0.0,
            dsr_prob=0.0,
            survivors_only=survivors_only,
        )
    denom = denom_inside / (T - 1)
    z = float((sr_hat - e_max_sr) / math.sqrt(denom))
    dsr_prob = float(stats.norm.cdf(z))
    return DsrResult(
        sharpe=sr_hat,
        n_trials=n_trials,
        sr_var_across_trials=sr_var_across_trials,
        e_max_sr=round(float(e_max_sr), 4),
        skew=round(skew, 4),
        kurt=round(kurt, 4),
        n_obs=n_obs,
        z=round(z, 4),
        dsr_prob=round(dsr_prob, 4),
        survivors_only=survivors_only,
    )


def sr_var_from_sweep(sweep_sharpes: list[float] | np.ndarray) -> float:
    """Sample variance of Sharpes across an entire sweep. The standard
    way to estimate `sr_var_across_trials`. Pass the FULL sweep, not
    just survivors (or pass survivors and set `survivors_only=True` in
    deflated_sharpe to flag the result as optimistic).
    """
    arr = np.asarray(sweep_sharpes, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    return float(np.var(arr, ddof=1))
