"""Sanctuary discipline -- standardised across the framework.

Specified in directives/Methodology Audit & Unified Framework 2026-05-14.md
§2.8. Fixes gaps C2 (inconsistent slicing) and C3 (divergence unexplained).

The sanctuary is the last N calendar months by `df.index[-1]`. It is:
    - Sliced BEFORE any IS/OOS fold construction.
    - Used for a ONE-SHOT final-validation pass after the WFO completes.
    - Tested for "luckiness" via a divergence test against historical
      rolling-12-month windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import pandas as pd

from titan.research.metrics import bootstrap_sharpe_ci, sharpe

T = TypeVar("T", pd.DataFrame, pd.Series)


@dataclass(frozen=True)
class SanctuarySlice:
    """Result of `slice_sanctuary`. Holds the visible portion + the
    held-out sanctuary, plus the exact boundary timestamp for audit logs.
    """

    visible: pd.DataFrame
    sanctuary: pd.DataFrame
    sanctuary_start: pd.Timestamp
    sanctuary_end: pd.Timestamp
    months_held_out: int


def slice_sanctuary(df: T, months: int = 12) -> SanctuarySlice:
    """Slice off the last ``months`` calendar months from a time-indexed df.

    Strict calendar-time slicing: `sanctuary_start = df.index[-1] - DateOffset(months)`.
    Returns a SanctuarySlice with both halves + the boundary timestamp.

    Raises TypeError if df.index isn't a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"slice_sanctuary requires DatetimeIndex, got {type(df.index).__name__}")
    if df.empty:
        raise ValueError("slice_sanctuary: empty input")
    end = df.index[-1]
    sanctuary_start = end - pd.DateOffset(months=months)
    visible = df[df.index < sanctuary_start]
    sanctuary = df[df.index >= sanctuary_start]
    return SanctuarySlice(
        visible=visible if isinstance(df, pd.DataFrame) else visible.to_frame(),
        sanctuary=sanctuary if isinstance(df, pd.DataFrame) else sanctuary.to_frame(),
        sanctuary_start=sanctuary_start,
        sanctuary_end=end,
        months_held_out=months,
    )


@dataclass(frozen=True)
class DivergenceTest:
    """Result of `sanctuary_divergence_test`.

    `sanctuary_sharpe` is the realised Sharpe on the held-out window.
    `historical_sharpe_distribution` is the full distribution of
    12-month-rolling-window Sharpes over the WFO visible portion.
    `percentile` is where the sanctuary Sharpe sits in that distribution
    (0.0 = lowest, 1.0 = highest).
    `lucky_flag` is True iff the sanctuary is in the top 5% of historical
    windows — interpretation: the strategy's recent performance may be
    regime-specific rather than deployment-validating.
    `unlucky_flag` is True iff the sanctuary is in the bottom 5% —
    interpretation: recent regime is adverse but historical edge is
    intact; not a deployment veto.
    """

    sanctuary_sharpe: float
    historical_sharpe_p5: float
    historical_sharpe_p25: float
    historical_sharpe_p50: float
    historical_sharpe_p75: float
    historical_sharpe_p95: float
    percentile: float
    lucky_flag: bool
    unlucky_flag: bool


def sanctuary_divergence_test(
    historical_returns: pd.Series,
    sanctuary_returns: pd.Series,
    periods_per_year: int,
    *,
    window_bars: int | None = None,
) -> DivergenceTest:
    """Test whether the sanctuary's Sharpe is unusually high or low vs
    the historical distribution of same-length rolling-window Sharpes.

    The test answers: "Is the recent 12-month Sharpe an outlier in the
    distribution of historical 12-month Sharpes?" If yes (top 5%), the
    sanctuary's positive Sharpe is likely regime-specific, not
    deployment-validating. This addresses the C3 gap from the audit
    catalogue: range-expansion / bond-equity / demo_fxmr all showed
    sanctuary >> historical OOS; this test quantifies how unusual that is.

    Parameters
    ----------
    historical_returns:
        Strategy per-bar (or per-day) returns over the WFO visible portion.
    sanctuary_returns:
        Strategy returns over the sanctuary window.
    periods_per_year:
        Annualisation factor matching the bar frequency of the returns.
    window_bars:
        Length of the rolling-window Sharpes used to build the historical
        distribution. Defaults to ``len(sanctuary_returns)`` so the
        comparison is apples-to-apples.
    """
    if window_bars is None:
        window_bars = max(20, len(sanctuary_returns))
    if len(historical_returns) < window_bars * 2:
        # Not enough history for a meaningful distribution.
        sh_sanc = sharpe(sanctuary_returns, periods_per_year=periods_per_year)
        return DivergenceTest(
            sanctuary_sharpe=sh_sanc,
            historical_sharpe_p5=float("nan"),
            historical_sharpe_p25=float("nan"),
            historical_sharpe_p50=float("nan"),
            historical_sharpe_p75=float("nan"),
            historical_sharpe_p95=float("nan"),
            percentile=float("nan"),
            lucky_flag=False,
            unlucky_flag=False,
        )

    # Build the historical distribution: every aligned rolling window.
    historical_window_sharpes: list[float] = []
    n = len(historical_returns)
    step = max(1, window_bars // 4)  # 75% overlap is fine for a distribution
    for start in range(0, n - window_bars, step):
        window = historical_returns.iloc[start : start + window_bars]
        sh = sharpe(window, periods_per_year=periods_per_year)
        if np.isfinite(sh):
            historical_window_sharpes.append(sh)

    if len(historical_window_sharpes) < 4:
        # Pathologically short series; can't compute percentile reliably.
        sh_sanc = sharpe(sanctuary_returns, periods_per_year=periods_per_year)
        return DivergenceTest(
            sanctuary_sharpe=sh_sanc,
            historical_sharpe_p5=float("nan"),
            historical_sharpe_p25=float("nan"),
            historical_sharpe_p50=float("nan"),
            historical_sharpe_p75=float("nan"),
            historical_sharpe_p95=float("nan"),
            percentile=float("nan"),
            lucky_flag=False,
            unlucky_flag=False,
        )

    arr = np.asarray(historical_window_sharpes)
    sh_sanc = sharpe(sanctuary_returns, periods_per_year=periods_per_year)
    pct = float((arr < sh_sanc).mean())
    return DivergenceTest(
        sanctuary_sharpe=float(round(sh_sanc, 4)),
        historical_sharpe_p5=float(round(np.quantile(arr, 0.05), 4)),
        historical_sharpe_p25=float(round(np.quantile(arr, 0.25), 4)),
        historical_sharpe_p50=float(round(np.quantile(arr, 0.50), 4)),
        historical_sharpe_p75=float(round(np.quantile(arr, 0.75), 4)),
        historical_sharpe_p95=float(round(np.quantile(arr, 0.95), 4)),
        percentile=round(pct, 4),
        lucky_flag=pct >= 0.95,
        unlucky_flag=pct <= 0.05,
    )


# --------------------------------------------------------------------------- #
# Multi-window / bootstrapped-CI sanctuary (P1-10)                             #
# --------------------------------------------------------------------------- #
#
# Specified in directives/Audit Remediation Plan 2026-05-29.md row P1-10:
# "multi-window / bootstrapped-CI sanctuary to replace the n=1 terminal
#  window; feed lucky_flag into decide() as a downgrade."
#
# A single terminal sanctuary window is one draw -- it can be lucky or unlucky
# purely by regime. Holding out the K most-recent disjoint windows and running
# the divergence test on EACH (against its own preceding history) turns the
# n=1 luck check into a K-of-K vote, and a bootstrapped CI on the pooled
# sanctuary Sharpe quantifies how firmly the held-out edge is positive.


@dataclass(frozen=True)
class MultiSanctuaryResult:
    """Aggregate of the multi-window sanctuary test.

    ``per_window`` holds the K divergence tests (most-recent first). The
    aggregate ``lucky_flag`` / ``unlucky_flag`` fire when a MAJORITY of the K
    windows are lucky / unlucky -- a robust signal that the held-out
    performance is regime-specific rather than deployment-validating. Feed
    ``lucky_flag`` into ``decision.decide(..., lucky_flag=...)`` for the
    one-level verdict downgrade.
    """

    n_windows: int
    months_per_window: int
    per_window: tuple[DivergenceTest, ...]
    window_sharpes: tuple[float, ...]
    pooled_sharpe: float
    sharpe_ci_lo: float
    sharpe_ci_hi: float
    n_lucky: int
    n_unlucky: int
    lucky_flag: bool
    unlucky_flag: bool


def slice_multi_sanctuary(df: T, *, n_windows: int = 3, months: int = 12) -> list[SanctuarySlice]:
    """Slice the ``n_windows`` most-recent disjoint ``months``-length windows.

    Returns a list of :class:`SanctuarySlice` (most-recent first). For each
    window, ``visible`` is everything strictly before that window's start, so
    each window can be divergence-tested against only its own prior history.
    Stops early if a window would be empty (insufficient history).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"slice_multi_sanctuary requires DatetimeIndex, got {type(df.index).__name__}"
        )
    if df.empty:
        raise ValueError("slice_multi_sanctuary: empty input")
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")

    end = df.index[-1]
    slices: list[SanctuarySlice] = []
    for k in range(n_windows):
        w_start = end - pd.DateOffset(months=months * (k + 1))
        w_end = end - pd.DateOffset(months=months * k)
        if k == 0:
            mask = df.index >= w_start  # include the final bar
        else:
            mask = (df.index >= w_start) & (df.index < w_end)
        sanctuary = df[mask]
        if sanctuary.empty:
            break
        visible = df[df.index < w_start]
        slices.append(
            SanctuarySlice(
                visible=visible if isinstance(df, pd.DataFrame) else visible.to_frame(),
                sanctuary=sanctuary if isinstance(df, pd.DataFrame) else sanctuary.to_frame(),
                sanctuary_start=w_start,
                sanctuary_end=w_end,
                months_held_out=months,
            )
        )
    return slices


def multi_window_sanctuary_test(
    strategy_returns: pd.Series,
    periods_per_year: int,
    *,
    n_windows: int = 3,
    months: int = 12,
    n_resamples: int = 1000,
    block_size: int | None = None,
    seed: int = 42,
    lucky_majority: int | None = None,
) -> MultiSanctuaryResult:
    """Run the divergence test on the K most-recent disjoint windows + a
    bootstrapped CI on the pooled sanctuary Sharpe.

    Parameters
    ----------
    strategy_returns:
        Full strategy per-bar return series (DatetimeIndex).
    periods_per_year:
        Annualisation factor matching the bar frequency.
    n_windows:
        Number of disjoint held-out windows K (most-recent first).
    months:
        Length of each window in calendar months.
    n_resamples, block_size, seed:
        Passed to :func:`titan.research.metrics.bootstrap_sharpe_ci` for the
        pooled-Sharpe CI. Pass ``block_size`` (stationary bootstrap) for
        serially-correlated returns so ``sharpe_ci_lo`` is not optimistic.
    lucky_majority:
        Number of lucky windows needed to set the aggregate ``lucky_flag``.
        Defaults to a strict majority ``floor(K/2) + 1``.

    Raises:
    ------
    ValueError
        If no non-empty sanctuary windows could be sliced.
    """
    slices = slice_multi_sanctuary(strategy_returns, n_windows=n_windows, months=months)
    if not slices:
        raise ValueError("multi_window_sanctuary_test: no non-empty sanctuary windows")

    per_window: list[DivergenceTest] = []
    pooled_parts: list[pd.Series] = []
    for sl in slices:
        sanc = sl.sanctuary.iloc[:, 0]
        hist = sl.visible.iloc[:, 0]
        per_window.append(sanctuary_divergence_test(hist, sanc, periods_per_year))
        pooled_parts.append(sanc)

    pooled = pd.concat(pooled_parts).sort_index()
    pooled_sharpe = float(round(sharpe(pooled, periods_per_year=periods_per_year), 4))
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        pooled,
        periods_per_year,
        n_resamples=n_resamples,
        seed=seed,
        block_size=block_size,
    )

    k = len(per_window)
    majority = lucky_majority if lucky_majority is not None else (k // 2 + 1)
    n_lucky = sum(1 for d in per_window if d.lucky_flag)
    n_unlucky = sum(1 for d in per_window if d.unlucky_flag)
    return MultiSanctuaryResult(
        n_windows=k,
        months_per_window=months,
        per_window=tuple(per_window),
        window_sharpes=tuple(d.sanctuary_sharpe for d in per_window),
        pooled_sharpe=pooled_sharpe,
        sharpe_ci_lo=float(round(ci_lo, 4)),
        sharpe_ci_hi=float(round(ci_hi, 4)),
        n_lucky=n_lucky,
        n_unlucky=n_unlucky,
        lucky_flag=n_lucky >= majority,
        unlucky_flag=n_unlucky >= majority,
    )
