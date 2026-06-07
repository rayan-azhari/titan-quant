"""Shared research & live math primitives.

Every Sharpe / vol / z-score / annualization in the codebase must go through
this module. The April 2026 audit found that identical calculations had been
reimplemented in at least six places, each with a different bug:

* ``research/auto/evaluate.py::_sharpe_from_rets`` — filtered ``rets != 0.0``
  before annualising with ``sqrt(252)``, overstating Sharpe by ``sqrt(1/P)``
  for a strategy that trades ``P%`` of days.
* ``research/auto/phase_portfolio.py::_sharpe`` — same bug.
* ``research/mean_reversion/run_confluence_regime_wfo.py::_sharpe`` — same.
* ``research/cross_asset/run_bond_equity_wfo.py::_daily_sharpe`` — same.
* ``research/portfolio/metrics.py`` — inlined ``sqrt(252)`` on what were
  sometimes H1 bar returns.
* Live sizing in ``titan/strategies/{demo_fxmr,gld_confluence,mr_fx,gap_fade}``
  — ``math.sqrt(var * 252)`` on H1/M5 returns, understating annualised vol
  by ~sqrt(24) (H1) to ~sqrt(288) (M5) and systematically over-sizing.

Design principles
-----------------
1.  ``periods_per_year`` is **required** on every Sharpe / vol call. There is
    no default. The bar timeframe determines the annualisation factor.
2.  No Sharpe function filters ``rets != 0`` internally. If the caller wants
    per-trade statistics they pass a trade-return series; otherwise Sharpe is
    computed over every bar in the input (zero-days included, because a flat
    day is information about the strategy's selectivity).
3.  All z-score helpers are explicitly either *causal rolling* or *IS-frozen*.
    Full-series ``(x - x.mean()) / x.std()`` is never offered — it is
    look-ahead by construction.
4.  Functions accept ``pd.Series`` or ``np.ndarray`` input and return the
    natural type for the output (scalar for Sharpe, Series for EWMA vol, etc).
5.  Numerical edge cases (empty series, constant series, NaN) return 0.0 or
    NaN rather than raising — these are explicitly documented per-function.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

# ── Annualisation constants ────────────────────────────────────────────────
#
# Bars per trading year per timeframe. These assume:
#   * 252 trading days per year (standard US convention).
#   * 24h FX / 6.5h equity day reflected only at the daily level — intraday
#     timeframes use the 24-hour clock because FX (the dominant intraday
#     dataset in this system) trades ~24h/day.
#
# For equity intraday strategies that actually only see 78 M5 bars per day
# (6.5h * 12), the caller must override with ``78 * 252`` rather than use
# the M5 value here. That divergence is the caller's responsibility because
# it is strategy-specific.

BARS_PER_YEAR: dict[str, int] = {
    "D": 252,
    "H4": 252 * 6,  # 6 H4 bars per FX day (24/4)
    "H1": 252 * 24,  # 24 H1 bars per FX day
    "M5": 252 * 24 * 12,  # 288 M5 bars per FX day
    "M1": 252 * 24 * 60,
}


def _as_series(x: pd.Series | np.ndarray | Iterable[float]) -> pd.Series:
    """Normalise input to a ``pd.Series`` with a clean RangeIndex if needed."""
    if isinstance(x, pd.Series):
        return x
    arr = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float)
    return pd.Series(arr)


# ── Sharpe ─────────────────────────────────────────────────────────────────


def sharpe(
    returns: pd.Series | np.ndarray | Iterable[float],
    periods_per_year: int,
    *,
    ddof: int = 1,
) -> float:
    """Annualised Sharpe ratio from a per-period return series.

    Parameters
    ----------

    Returns:
        Per-period returns (decimal, not percent). May be a Series, ndarray,
        or iterable. ``NaN`` values are dropped. **Zero-return bars are kept**
        — filtering them biases the estimator upward by ``sqrt(1/P)`` when
        the strategy trades only ``P%`` of bars.
    periods_per_year:
        Annualisation factor. ``252`` for daily, ``252*24`` for H1, etc.
        Use :data:`BARS_PER_YEAR` or a strategy-specific override.
    ddof:
        Degrees of freedom for std (default 1, matches pandas / numpy
        sample-std convention).

    Returns:
    -------
    float
        Annualised Sharpe. Returns ``0.0`` if fewer than 20 non-NaN samples
        or if std is below ``1e-12`` (constant series).
    """
    s = _as_series(returns).dropna()
    if len(s) < 20:
        return 0.0
    sd = float(s.std(ddof=ddof))
    if sd < 1e-12:
        return 0.0
    return float(s.mean() / sd * math.sqrt(periods_per_year))


def trade_sharpe(
    trade_returns: pd.Series | np.ndarray | Iterable[float],
    trades_per_year: float,
    *,
    ddof: int = 1,
) -> float:
    """Sharpe on a *per-trade* return series, annualised by ``trades_per_year``.

    Use this when you actually have trade-level P&L. The per-bar
    :func:`sharpe` treats zero-return bars as information about selectivity;
    this function presumes every element is a completed trade.
    """
    s = _as_series(trade_returns).dropna()
    if len(s) < 10:
        return 0.0
    sd = float(s.std(ddof=ddof))
    if sd < 1e-12:
        return 0.0
    return float(s.mean() / sd * math.sqrt(trades_per_year))


# ── Vol / annualisation ────────────────────────────────────────────────────


def annualize_vol(per_period_std: float, periods_per_year: int) -> float:
    """Scale a per-period std to an annual vol.

    ``annualize_vol(0.01, 252) == 0.01 * sqrt(252)`` ≈ 0.1587 (15.9% annual
    vol from 1%/day).
    """
    if per_period_std < 0 or per_period_std != per_period_std:  # NaN guard
        return 0.0
    return float(per_period_std * math.sqrt(periods_per_year))


def ewm_vol(
    returns: pd.Series | np.ndarray | Iterable[float],
    lam: float = 0.94,
    *,
    periods_per_year: int,
    min_periods: int = 10,
) -> pd.Series:
    """EWMA realised vol series, annualised.

    Uses the RiskMetrics-style recursion ``var_t = lam * var_{t-1} +
    (1-lam) * r_t^2`` with pandas ``ewm(alpha=1-lam)`` semantics.

    Parameters
    ----------

    Returns:
        Per-period returns. Input is normalised to a Series; NaNs are
        propagated (they become NaN in the output).
    lam:
        Decay factor (RiskMetrics default 0.94).
    periods_per_year:
        Required keyword — annualisation factor matching the bar frequency.
    min_periods:
        The output is NaN until this many bars have accumulated.

    Returns:
    -------
    pd.Series
        Annualised vol per bar, same index as input.
    """
    if not 0 < lam < 1:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    s = _as_series(returns)
    var = s.pow(2).ewm(alpha=1.0 - lam, adjust=False, min_periods=min_periods).mean()
    return var.pow(0.5) * math.sqrt(periods_per_year)


def ewm_vol_last(
    returns: pd.Series | np.ndarray | Iterable[float],
    lam: float = 0.94,
    *,
    periods_per_year: int,
    min_periods: int = 10,
) -> float:
    """Scalar convenience: last value of :func:`ewm_vol`. ``0.0`` if insufficient.

    Used by live sizing code that wants a single annualised vol number for
    the current bar.
    """
    s = ewm_vol(returns, lam=lam, periods_per_year=periods_per_year, min_periods=min_periods)
    if s.empty:
        return 0.0
    last = s.iloc[-1]
    if pd.isna(last):
        return 0.0
    return float(last)


# ── Z-score helpers (causal only) ──────────────────────────────────────────


def rolling_zscore(
    x: pd.Series | np.ndarray | Iterable[float],
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.Series:
    """Causal rolling z-score: ``(x_t - mean(x[t-w+1..t])) / std(x[t-w+1..t])``.

    The value at bar ``t`` depends only on observations through ``t``.
    No leakage. Use this instead of global ``(x - x.mean()) / x.std()``.
    """
    s = _as_series(x)
    mp = min_periods if min_periods is not None else max(2, window // 4)
    mean = s.rolling(window=window, min_periods=mp).mean()
    std = s.rolling(window=window, min_periods=mp).std(ddof=ddof)
    return ((s - mean) / std.replace(0.0, np.nan)).astype(float)


def expanding_zscore(
    x: pd.Series | np.ndarray | Iterable[float],
    *,
    min_periods: int = 2,
    ddof: int = 1,
) -> pd.Series:
    """Causal expanding z-score: ``(x_t - mean(x[..t])) / std(x[..t])``.

    The value at bar ``t`` uses only observations through ``t`` (expanding
    window) -- no leakage. Centralises the ``(x - x.expanding().mean()) /
    x.expanding().std()`` idiom so it lives in one audited place (used by the
    correlation-dial leverage governor).
    """
    s = _as_series(x)
    mean = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std(ddof=ddof)
    return ((s - mean) / std.replace(0.0, np.nan)).astype(float)


def is_frozen_zscore(
    x: pd.Series | np.ndarray | Iterable[float],
    is_end_idx: int,
    *,
    ddof: int = 1,
) -> pd.Series:
    """Z-score using mean/std computed only on ``x[:is_end_idx]``.

    Matches the WFO convention of freezing normalisation stats at the IS /
    OOS boundary. The whole series (IS and OOS) is then z-scored with those
    frozen stats, so OOS bars see no OOS statistics.
    """
    s = _as_series(x)
    if is_end_idx <= 0 or is_end_idx > len(s):
        raise ValueError(f"is_end_idx out of range: {is_end_idx} (len={len(s)})")
    is_slice = s.iloc[:is_end_idx].dropna()
    if len(is_slice) < 2:
        return pd.Series(np.nan, index=s.index)
    mu = float(is_slice.mean())
    sd = float(is_slice.std(ddof=ddof))
    if sd < 1e-12:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


# ── Drawdown / related ─────────────────────────────────────────────────────


def max_drawdown(returns: pd.Series | np.ndarray | Iterable[float]) -> float:
    """Max drawdown of the cumulative-return curve. Returns a non-positive float.

    Convention: the equity curve is anchored at 1.0 before the first return,
    so a first-bar loss is a real drawdown from 1.0 (not zero because the
    peak was never below 1.0). A naive ``(1+r).cumprod().cummax()`` misses
    this and under-reports drawdowns that start on bar 0.

    ``max_drawdown([]) == 0.0``. Missing values are treated as 0.
    """
    s = _as_series(returns).fillna(0.0)
    if len(s) < 2:
        return 0.0
    # Prepend the starting capital (1.0) so the peak history includes it.
    eq = pd.concat([pd.Series([1.0]), (1.0 + s).cumprod().reset_index(drop=True)])
    dd = (eq - eq.cummax()) / eq.cummax()
    return float(dd.min())


def cvar(
    returns: pd.Series | np.ndarray | Iterable[float],
    *,
    alpha: float = 0.05,
) -> float:
    """Conditional Value-at-Risk (Expected Shortfall) at level alpha.

    CVaR_alpha is the average of returns in the worst `alpha` fraction of
    outcomes. More robust than VaR for tail risk: captures the average
    severity of bad days, not just the threshold.

    Convention: returns negative numbers for typical loss distributions
    (since losses live in the left tail). `cvar([0.01]*100 + [-0.10])`
    with alpha=0.05 averages the 5 worst outcomes.

    Reference: Rockafellar & Uryasev 2000, "Optimization of CVaR".
    """
    s = _as_series(returns).dropna()
    n = len(s)
    if n < 20:
        return 0.0
    k = max(1, int(np.ceil(n * alpha)))
    worst = np.sort(s.to_numpy())[:k]
    return float(worst.mean())


def cdar(
    returns: pd.Series | np.ndarray | Iterable[float],
    *,
    alpha: float = 0.05,
) -> float:
    """Conditional Drawdown-at-Risk at level alpha.

    Average of the worst `alpha` fraction of drawdown depths along the
    equity curve. Captures "many medium drawdowns" patterns that MaxDD
    misses by reporting only a single point.

    Reference: Chekhlov, Uryasev & Zabarankin 2005, "Drawdown Measure
    in Portfolio Optimization".
    """
    s = _as_series(returns).fillna(0.0)
    n = len(s)
    if n < 20:
        return 0.0
    eq = pd.concat([pd.Series([1.0]), (1.0 + s).cumprod().reset_index(drop=True)])
    dd = (eq - eq.cummax()) / eq.cummax()
    dd_arr = dd.to_numpy()
    k = max(1, int(np.ceil(n * alpha)))
    worst = np.sort(dd_arr)[:k]  # most-negative values first
    return float(worst.mean())


def calmar(
    returns: pd.Series | np.ndarray | Iterable[float],
    periods_per_year: int,
) -> float:
    """Calmar ratio: annualised return / |max drawdown|. Returns 0 if dd is zero."""
    s = _as_series(returns).dropna()
    if len(s) < 20:
        return 0.0
    ann_ret = float(s.mean() * periods_per_year)
    dd = abs(max_drawdown(s))
    return ann_ret / dd if dd > 1e-9 else 0.0


def sortino(
    returns: pd.Series | np.ndarray | Iterable[float],
    periods_per_year: int,
    *,
    target: float = 0.0,
    ddof: int = 1,
) -> float:
    """Sortino ratio: annualised excess-over-target divided by downside std."""
    s = _as_series(returns).dropna()
    if len(s) < 20:
        return 0.0
    downside = s[s < target]
    if len(downside) < 5:
        return 0.0
    dsd = float(downside.std(ddof=ddof))
    if dsd < 1e-12:
        return 0.0
    return float((s.mean() - target) / dsd * math.sqrt(periods_per_year))


# ── Bootstrap CIs ──────────────────────────────────────────────────────────


def _stationary_bootstrap_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis & Romano (1994) stationary-bootstrap index path of length ``n``.

    At each step, with probability ``p = 1/mean_block`` start a new block at a
    random index; otherwise continue the previous block (circularly). Block
    lengths are geometric with mean ``mean_block``, so serial dependence up to
    ~that scale is preserved -- unlike the IID bootstrap which destroys it.
    """
    p = 1.0 / max(1.0, float(mean_block))
    new_block = rng.random(n) < p
    new_block[0] = True
    random_starts = rng.integers(0, n, size=n)
    idx = np.empty(n, dtype=np.int64)
    cur = 0
    for t in range(n):
        cur = int(random_starts[t]) if new_block[t] else (cur + 1) % n
        idx[t] = cur
    return idx


def bootstrap_sharpe_ci(
    returns: pd.Series | np.ndarray | Iterable[float],
    periods_per_year: int,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    *,
    seed: int = 42,
    block_size: int | None = None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for :func:`sharpe` (percentile method).

    Two modes (audit P1-7, External Quant Audit 2026-05-29):

    - ``block_size is None`` (default): IID bootstrap. Simple, but it DESTROYS
      serial correlation, so for autocorrelated returns (trend / carry) it
      narrows the CI and biases ``CI_lo`` UPWARD -- the optimism the audit
      flagged on the CI_lo decision axis. Kept as the default only for
      backward compatibility.
    - ``block_size`` given: **stationary bootstrap** (Politis & Romano 1994)
      with geometric mean block length ``block_size``, which preserves serial
      dependence and yields an honest (wider) CI. Audit callers / ``run_audit``
      pass the strategy class's ``McConfig.block_size_bars`` here.

    Parameters
    ----------

    Returns:
        Return series (same semantics as :func:`sharpe`).
    periods_per_year:
        Required annualisation factor.
    n_resamples:
        Number of bootstrap resamples (default 1000).
    confidence:
        Two-sided confidence level (default 0.95).
    seed:
        RNG seed for reproducibility.
    block_size:
        If given, use the stationary bootstrap with this geometric mean block
        length (serially-aware). If ``None``, IID.

    Returns:
    -------
    (lo, hi) : tuple[float, float]
        The ``(confidence)`` two-sided CI on the annualised Sharpe.
        Returns ``(0.0, 0.0)`` if fewer than 20 samples.
    """
    s = _as_series(returns).dropna().to_numpy()
    n = len(s)
    if n < 20:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    sharpes = np.empty(n_resamples)
    sqrt_ann = math.sqrt(periods_per_year)
    for i in range(n_resamples):
        if block_size is None:
            resample = rng.choice(s, size=n, replace=True)
        else:
            resample = s[_stationary_bootstrap_indices(n, block_size, rng)]
        sd = resample.std(ddof=1)
        sharpes[i] = 0.0 if sd < 1e-12 else resample.mean() / sd * sqrt_ann
    lo = float(np.quantile(sharpes, alpha / 2.0))
    hi = float(np.quantile(sharpes, 1.0 - alpha / 2.0))
    return (lo, hi)
