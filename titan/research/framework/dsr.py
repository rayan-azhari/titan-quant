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
    periods_per_year: int = 252,
    survivors_only: bool = False,
) -> DsrResult:
    """Bailey & López de Prado 2014 DSR with skew + kurt from the actual
    return distribution (no normal-returns assumption).

    FREQUENCY CONTRACT (2026-07-15 units-bug fix). The DSR z-statistic is a Sharpe
    divided by the standard error of that SAME Sharpe, and the SE is
    ``sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1))`` where ``T`` is the number of
    return OBSERVATIONS. So the Sharpe in the z and the ``T`` in the SE must be at the
    SAME frequency. ``sr_hat`` / ``sr_var_across_trials`` are supplied ANNUALISED (to
    match every other Sharpe in the codebase and the deployment gate), but ``returns``
    is a per-observation series, so this function converts the annualised Sharpe and the
    annualised expected-max to PER-OBSERVATION (divide by ``sqrt(periods_per_year)``)
    before forming z. Feeding an annualised Sharpe straight into a ``/(T-1)`` that counts
    daily bars inflated z by ~``sqrt(periods_per_year)`` (~16x for daily) and pinned
    ``dsr_prob`` near 1.000 -- i.e. the ``dsr_prob >= 0.95`` deployment gate was
    near-unfailable. Reported ``sharpe`` / ``e_max_sr`` stay ANNUALISED for readability;
    only the z math is per-observation.

    Parameters
    ----------
    sr_hat:
        Observed ANNUALISED Sharpe.
    sr_var_across_trials:
        Variance of the (annualised) Sharpes across the FULL sweep. If only survivors are
        available, pass `survivors_only=True` -- the result is flagged as OPTIMISTIC
        (true variance is larger, true DSR is worse).

    Returns:
        This trial's per-observation return series. Skew + kurt are estimated from this;
        its length is ``T``.
    n_trials:
        Total cells in the sweep (NOT survivors).
    periods_per_year:
        Bar frequency of ``returns`` used to de-annualise ``sr_hat`` for the z math.
        252 for daily (default); pass 252*24 for hourly, 12 for monthly, etc. -- it MUST
        match the annualisation used to produce ``sr_hat``.

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
    # Use the REAL observation count for the SE. A prior `max(n_obs, 30)` fabricated
    # observations for tiny samples, shrinking the SE and biasing the gate toward PASS
    # (the wrong direction). For real daily strategies n_obs >> 30 so this is a no-op;
    # for a tiny sample the true (small) T correctly widens the SE -> harder to pass.
    T = n_obs
    # Bailey & López de Prado 2014 eq. (4). Variance-stabilised gap z is:
    #   z = (SR - E[maxSR]) / sqrt( (1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1) )
    # where SR is the PER-OBSERVATION Sharpe (same frequency as the T observations). The
    # supplied sr_hat / e_max_sr are annualised -> de-annualise them here (see the
    # frequency-contract note in the docstring; annualised-SR-into-daily-T was the bug).
    sqrt_ppy = math.sqrt(periods_per_year)
    sr_hat_obs = sr_hat / sqrt_ppy
    e_max_obs = e_max_sr / sqrt_ppy
    denom_inside = 1.0 - skew * sr_hat_obs + (kurt - 1.0) / 4.0 * sr_hat_obs**2
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
    z = float((sr_hat_obs - e_max_obs) / math.sqrt(denom))
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
