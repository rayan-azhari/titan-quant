"""Calmar primitive — geometric CAGR, MaxDD, Calmar ratio, lift, bootstrap CI,
and the V3.8 promotion gate.

Per `directives/Objective Reframe 2026-05-23.md` §2.2:

    Calmar lift replaces Sharpe lift as the primary L67 promotion metric.
    Sharpe lift is retained as a secondary informational metric.

V3.8 promotion gate (from the directive):

    - Calmar lift >= +0.10 vs current portfolio (1y rolling) AND
    - Sharpe lift >= 0 vs current portfolio (no regression on V3.7 metric) AND
    - P_kill (joint) <= 1e-3 AND
    - P(MaxDD > 25%) (joint) <= 5%.

This module implements the first two conditions. The L65 P_kill and MC
P(MaxDD) conditions remain in `ruin.py` and `mc.py` respectively; promotion
verdict combines all four at the orchestration layer.

Math
====

CAGR is the GEOMETRIC annualised return — distinct from the arithmetic
`mean * periods_per_year` used in `titan.research.metrics.calmar`. For
strategies with non-trivial volatility the two differ materially (arithmetic
> geometric by ~vol^2 / 2 over long horizons) and only the geometric form
correctly compounds across a multi-year audit window.

    eq[t]   = (1 + r[1]) * (1 + r[2]) * ... * (1 + r[t])
    CAGR    = eq[-1] ** (periods_per_year / n_bars) - 1
    MaxDD   = min over t of (eq[t] - cummax(eq))[t] / cummax(eq)[t]
    Calmar  = CAGR / |MaxDD|

The MaxDD path uses `titan.research.metrics.max_drawdown` for consistency
with the rest of the framework (which anchors the equity curve at 1.0
before bar 0 — see metrics.py docstring).

Bootstrap CI
============

IID block-of-1 bootstrap, percentile method, matching the convention in
`titan.research.metrics.bootstrap_sharpe_ci`. For serially-correlated
returns the percentile CI is a low-bound estimate of true uncertainty;
this is acceptable for the "does Calmar lift include zero?" gate but the
caller should use `mc.run_block_mc` for the joint ruin / DD constraints.

Lift gates and edge cases
=========================

- Empty / sub-20-bar series return Calmar = 0.0 by metrics convention.
- Zero-drawdown series (all returns >= 0) return Calmar = 0.0 to avoid
  division-by-zero; in practice this only occurs on synthetic tests.
- `calmar_lift` on a candidate proposed AND current both with Calmar = 0
  returns 0.0 (no lift to evaluate); promotion gate FAILS the lift check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from titan.research.metrics import max_drawdown, sharpe

# V3.8 §2.2 promotion gate defaults.
DEFAULT_CALMAR_LIFT_GATE: float = 0.10
DEFAULT_SHARPE_LIFT_GATE: float = 0.0

# Minimum bar count before Calmar is considered meaningful. Matches the
# 20-bar threshold used in `titan.research.metrics.calmar` and `sortino`.
MIN_BARS_FOR_CALMAR: int = 20


@dataclass(frozen=True)
class CalmarResult:
    """Point estimate of Calmar + its components.

    Attributes:
    ----------
    calmar:
        CAGR / |MaxDD|. Returns 0.0 if MaxDD is zero (no drawdown
        recorded — usually only on synthetic monotone-up tests).
    cagr:
        Geometric annualised return: ``eq[-1] ** (ppy / n) - 1``.
    max_dd:
        Max drawdown as a non-positive float (e.g. -0.15 = 15% DD).
    n_bars:
        Number of non-NaN bars in the input series.
    """

    calmar: float
    cagr: float
    max_dd: float
    n_bars: int


@dataclass(frozen=True)
class CalmarCi:
    """Bootstrap confidence interval for Calmar.

    Attributes:
    ----------
    point:
        Calmar of the input series (not the bootstrap mean — point
        estimate matches `compute_calmar(...).calmar`).
    ci_lo, ci_hi:
        Percentile bounds at the requested confidence level.
    confidence:
        Confidence level (e.g. 0.95 = 95% CI).
    n_resamples:
        Number of IID bootstrap resamples drawn.
    """

    point: float
    ci_lo: float
    ci_hi: float
    confidence: float
    n_resamples: int


@dataclass(frozen=True)
class CalmarPromotionResult:
    """Verdict of the V3.8 §2.2 promotion gate.

    The gate combines Calmar lift (primary) AND Sharpe lift (secondary,
    anti-regression) into a single PASS/FAIL. Either failing means the
    candidate is not promotion-eligible on these two metrics alone — the
    full promotion decision additionally requires L65 ruin + MC DD gates
    cleared elsewhere.

    Attributes:
    ----------
    current_calmar, proposed_calmar:
        Calmar point estimates for the two portfolios being compared.
    current_sharpe, proposed_sharpe:
        Sharpe point estimates for the same two portfolios.
    calmar_lift:
        ``proposed_calmar.calmar - current_calmar.calmar``.
    sharpe_lift:
        ``proposed_sharpe - current_sharpe``.
    passes_calmar_gate:
        True iff ``calmar_lift >= calmar_lift_gate``.
    passes_sharpe_gate:
        True iff ``sharpe_lift >= sharpe_lift_gate``.
    passes:
        True iff BOTH gates pass.
    reasons:
        Human-readable explanation of any failing gate. Empty tuple on
        full PASS.
    """

    current_calmar: CalmarResult
    proposed_calmar: CalmarResult
    current_sharpe: float
    proposed_sharpe: float
    calmar_lift: float
    sharpe_lift: float
    calmar_lift_gate: float
    sharpe_lift_gate: float
    passes_calmar_gate: bool
    passes_sharpe_gate: bool
    passes: bool
    reasons: tuple[str, ...]


def compute_cagr(
    returns: pd.Series | np.ndarray | Iterable[float],
    periods_per_year: int,
) -> float:
    """Geometric annualised return.

    ``CAGR = (eq[-1] / eq[0]) ** (ppy / n) - 1``

    Distinct from `mean(returns) * periods_per_year` (arithmetic ann
    return), which overstates compounding returns by ~vol^2 / 2 per year
    for typical equity strategies. Use this for Calmar / Calmar lift.

    Returns 0.0 if the series has fewer than `MIN_BARS_FOR_CALMAR`
    non-NaN bars.
    """
    s = pd.Series(returns).dropna()
    n = len(s)
    if n < MIN_BARS_FOR_CALMAR:
        return 0.0
    eq_final = float((1.0 + s).prod())
    if eq_final <= 0.0:
        # Total wipe-out (eq <= 0). Caller should treat as catastrophic.
        return -1.0
    return eq_final ** (periods_per_year / n) - 1.0


def compute_calmar(
    returns: pd.Series | np.ndarray | Iterable[float],
    periods_per_year: int,
) -> CalmarResult:
    """Calmar = CAGR / |MaxDD| with the supporting components.

    Returns Calmar = 0.0 when MaxDD == 0 (no drawdown recorded). This
    avoids division-by-zero on synthetic monotone-up series and matches
    the convention in `titan.research.metrics.calmar`.
    """
    s = pd.Series(returns).dropna()
    n = int(len(s))
    if n < MIN_BARS_FOR_CALMAR:
        return CalmarResult(calmar=0.0, cagr=0.0, max_dd=0.0, n_bars=n)
    cagr = compute_cagr(s, periods_per_year)
    mdd = float(max_drawdown(s))
    if abs(mdd) < 1e-9:
        return CalmarResult(calmar=0.0, cagr=cagr, max_dd=mdd, n_bars=n)
    return CalmarResult(calmar=cagr / abs(mdd), cagr=cagr, max_dd=mdd, n_bars=n)


def bootstrap_calmar_ci(
    returns: pd.Series | np.ndarray | Iterable[float],
    periods_per_year: int,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    *,
    seed: int = 42,
) -> CalmarCi:
    """IID bootstrap percentile CI for Calmar.

    Uses block-of-1 (IID) bootstrap, matching the convention in
    `titan.research.metrics.bootstrap_sharpe_ci`. For serially-correlated
    returns the CI underestimates true uncertainty; this is acceptable
    for the V3.8 "Calmar lift > 0?" gate but `mc.run_block_mc` should be
    used for the joint MC DD / ruin constraints elsewhere.

    Parameters
    ----------

    Returns:
        Per-bar returns. NaNs dropped before resampling.
    periods_per_year:
        Annualisation factor matching the bar frequency. Required (no
        default) per the research-math guardrail (L60).
    n_resamples:
        Number of bootstrap resamples. 1000 is the metrics-module
        default; raise to 2000 for tighter CIs at audit time.
    confidence:
        e.g. 0.95 for 95% CI. Symmetric percentile bounds.
    seed:
        RNG seed for reproducibility (matches metrics-module convention).

    Returns:
    -------
    `CalmarCi` with point estimate, lo/hi bounds, and metadata.
    """
    s = pd.Series(returns).dropna().to_numpy()
    n = len(s)
    point = compute_calmar(s, periods_per_year).calmar
    if n < MIN_BARS_FOR_CALMAR:
        return CalmarCi(
            point=point,
            ci_lo=point,
            ci_hi=point,
            confidence=confidence,
            n_resamples=0,
        )

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        c = compute_calmar(s[idx], periods_per_year).calmar
        samples.append(c)

    arr = np.asarray(samples)
    alpha = (1.0 - confidence) / 2.0
    ci_lo = float(np.quantile(arr, alpha))
    ci_hi = float(np.quantile(arr, 1.0 - alpha))
    return CalmarCi(
        point=point,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        confidence=confidence,
        n_resamples=n_resamples,
    )


def calmar_lift(
    proposed_returns: pd.Series | np.ndarray | Iterable[float],
    current_returns: pd.Series | np.ndarray | Iterable[float],
    periods_per_year: int,
) -> float:
    """Calmar(proposed) - Calmar(current). Positive == proposed is better.

    Computed on the FULL series passed in (typically the joint visible
    window after sanctuary slicing). Caller is responsible for aligning
    the two series to the same index if they differ.
    """
    cp = compute_calmar(proposed_returns, periods_per_year)
    cc = compute_calmar(current_returns, periods_per_year)
    return cp.calmar - cc.calmar


def evaluate_promotion(
    proposed_returns: pd.Series | np.ndarray | Iterable[float],
    current_returns: pd.Series | np.ndarray | Iterable[float],
    periods_per_year: int,
    *,
    calmar_lift_gate: float = DEFAULT_CALMAR_LIFT_GATE,
    sharpe_lift_gate: float = DEFAULT_SHARPE_LIFT_GATE,
) -> CalmarPromotionResult:
    """Apply the V3.8 §2.2 Calmar + Sharpe promotion gate.

    Returns a `CalmarPromotionResult` whose `.passes` is True only when
    BOTH the Calmar lift gate (default +0.10) AND the Sharpe-lift
    no-regression gate (default >= 0) are satisfied.

    This is the FRAMEWORK-LEVEL gate from V3.8 doctrine §2.2. Full
    promotion eligibility additionally requires the joint L65 (ruin) and
    MC (P(MaxDD > 25%)) gates cleared elsewhere — see `ruin.py` and
    `mc.py`. The orchestrator (typically `decide(DecisionInputs(...))`)
    combines all four.
    """
    proposed_calmar = compute_calmar(proposed_returns, periods_per_year)
    current_calmar = compute_calmar(current_returns, periods_per_year)
    proposed_sharpe = float(sharpe(proposed_returns, periods_per_year=periods_per_year))
    current_sharpe = float(sharpe(current_returns, periods_per_year=periods_per_year))

    cal_lift = proposed_calmar.calmar - current_calmar.calmar
    shp_lift = proposed_sharpe - current_sharpe

    passes_cal = cal_lift >= calmar_lift_gate
    passes_shp = shp_lift >= sharpe_lift_gate

    reasons: list[str] = []
    if not passes_cal:
        reasons.append(
            f"Calmar lift {cal_lift:+.4f} below gate "
            f"+{calmar_lift_gate:.2f} (current {current_calmar.calmar:+.4f}, "
            f"proposed {proposed_calmar.calmar:+.4f})"
        )
    if not passes_shp:
        reasons.append(
            f"Sharpe lift {shp_lift:+.4f} below gate "
            f"{sharpe_lift_gate:+.2f} (current {current_sharpe:+.4f}, "
            f"proposed {proposed_sharpe:+.4f}) -- V3.7 regression"
        )

    return CalmarPromotionResult(
        current_calmar=current_calmar,
        proposed_calmar=proposed_calmar,
        current_sharpe=current_sharpe,
        proposed_sharpe=proposed_sharpe,
        calmar_lift=cal_lift,
        sharpe_lift=shp_lift,
        calmar_lift_gate=calmar_lift_gate,
        sharpe_lift_gate=sharpe_lift_gate,
        passes_calmar_gate=passes_cal,
        passes_sharpe_gate=passes_shp,
        passes=passes_cal and passes_shp,
        reasons=tuple(reasons),
    )


__all__ = [
    "DEFAULT_CALMAR_LIFT_GATE",
    "DEFAULT_SHARPE_LIFT_GATE",
    "MIN_BARS_FOR_CALMAR",
    "CalmarResult",
    "CalmarCi",
    "CalmarPromotionResult",
    "compute_cagr",
    "compute_calmar",
    "bootstrap_calmar_ci",
    "calmar_lift",
    "evaluate_promotion",
]
