"""Crisis-regime detector (V3.8 §4.6 C3).

Per `directives/Objective Reframe 2026-05-23.md` §4.6:

    C3 -- Crisis-regime heat reduction: VIX > 30 OR realised 5d vol > 90th
    percentile (rolling 5y, per-instrument). Portfolio heat cap drops
    8% -> 6%; gross leverage cap drops 8x -> 6x; Kelly fractions stay at
    §4.3 setting.

    Rationale: a regime-conditional version of §4.3. The DD throttle reacts
    to losses already realised; C3 reacts to forward-looking risk (VIX or
    realised vol) BEFORE losses materialise. Roughly halves the C1
    worst-case (16% -> 11% in the same scenario).

This module is a pure function over (vix_value, vol_percentile). The PRM
`on_bar` hook computes / supplies both inputs and uses the result to
toggle `regime_normal` on the leverage envelope and portfolio heat
checks.

Why separate inputs (not derive vol_percentile internally)
==========================================================

The 90th-percentile-over-rolling-5y computation needs 5 years of
per-instrument vol history. Different callers have that history at
different stages of their pipeline:
- Live PRM: maintains rolling state per instrument, computes once.
- Backtest harness: computes per-bar via vectorised pandas ops.
- Audit script: may pre-load and freeze the percentile distribution.

Forcing the regime detector to compute the percentile would couple it to
one pipeline shape. Instead the detector takes the percentile (0-1) as a
NUMBER and asks the caller to compute it however suits. Helpers
`compute_realised_vol_annualised` + `compute_vol_percentile_current` are
provided for the common paths.

OR semantics (not AND)
======================

The detector OR-combines the two triggers: crisis if EITHER VIX > 30 OR
vol percentile > 0.90. The rationale per §4.6.4: VIX captures market-wide
fear (equity-sleeve specific), vol percentile captures per-instrument
stress (works on bonds, FX, commodities where VIX is not informative).
Either signal alone is sufficient to engage the crisis-regime de-risking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# V3.8 §4.6 C3 defaults.
DEFAULT_VIX_THRESHOLD: float = 30.0
DEFAULT_VOL_PERCENTILE_THRESHOLD: float = 0.90
DEFAULT_VOL_WINDOW_BARS: int = 5  # 5-day realised vol per spec.
DEFAULT_VOL_PERCENTILE_WINDOW_BARS: int = 252 * 5  # 5y of daily bars.


@dataclass(frozen=True)
class RegimeResult:
    """Crisis-regime evaluation snapshot.

    Returned by `is_crisis_regime`. Consumed by the PRM `on_bar` hook to
    toggle `regime_normal` on the leverage_envelope and portfolio_heat
    checks for the current bar.

    Attributes:
    ----------
    is_crisis:
        True iff at least one of the two triggers fired. Drives the
        leverage / heat / Kelly de-risking per §4.6 C3.
    vix_value:
        Latest VIX spot reading. None means VIX data was not supplied.
    vix_threshold:
        VIX level at which the VIX trigger fires (default 30.0).
    vix_triggered:
        True iff `vix_value is not None and vix_value > vix_threshold`.
    realised_vol_pct:
        Percentile (0-1) of the current realised-vol reading within its
        trailing 5y history. None means insufficient history.
    realised_vol_pct_threshold:
        Percentile at which the realised-vol trigger fires (default 0.90).
    realised_vol_triggered:
        True iff `realised_vol_pct > realised_vol_pct_threshold`.
    reasons:
        Human-readable explanation of which trigger fired. Empty tuple
        when `is_crisis is False`.
    """

    is_crisis: bool
    vix_value: float | None
    vix_threshold: float
    vix_triggered: bool
    realised_vol_pct: float | None
    realised_vol_pct_threshold: float
    realised_vol_triggered: bool
    reasons: tuple[str, ...]


def compute_realised_vol_annualised(
    returns: pd.Series | np.ndarray | Iterable[float],
    *,
    window_bars: int = DEFAULT_VOL_WINDOW_BARS,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling annualised realised vol (std × sqrt(ppy)) on per-bar returns.

    Returns a series of the same length as `returns`, with the first
    `window_bars - 1` values NaN (no full window yet). Caller is
    responsible for matching `periods_per_year` to the bar timeframe
    per L60 (no default 252 hidden assumption).

    For the V3.8 §4.6 C3 use case the spec is 5-day realised vol on
    daily bars, so the default `window_bars=5, periods_per_year=252`
    matches directly.
    """
    s = pd.Series(returns)
    return s.rolling(window_bars, min_periods=window_bars).std() * math.sqrt(periods_per_year)


def compute_vol_percentile_current(
    vol_history: pd.Series | np.ndarray | Iterable[float],
    *,
    lookback_bars: int = DEFAULT_VOL_PERCENTILE_WINDOW_BARS,
) -> float | None:
    """Percentile (0-1) of the LATEST vol reading within its trailing window.

    Walks back `lookback_bars` from the end of `vol_history`, drops NaNs,
    and computes the rank of the last value as `rank / N`. Returns None
    if fewer than 60 non-NaN observations are available in the window
    (matches the framework's general "60 bars before trusting a
    statistic" convention).

    For the V3.8 default of 5y rolling window on daily bars
    (1260 bars), most live deployments will have sufficient history
    once the framework has been running for ~3 months.
    """
    s = pd.Series(vol_history).dropna()
    if len(s) == 0:
        return None
    window = s.iloc[-lookback_bars:] if len(s) > lookback_bars else s
    if len(window) < 60:
        return None
    latest = float(window.iloc[-1])
    # Rank percentile: count strictly-less-than + half of equal, divided by N.
    n = len(window)
    less_than = (window < latest).sum()
    equal = (window == latest).sum()
    rank = (less_than + 0.5 * equal) / n
    return float(rank)


def is_crisis_regime(
    vix_value: float | None,
    realised_vol_pct: float | None,
    *,
    vix_threshold: float = DEFAULT_VIX_THRESHOLD,
    vol_percentile_threshold: float = DEFAULT_VOL_PERCENTILE_THRESHOLD,
) -> RegimeResult:
    """OR-combine the VIX and realised-vol-percentile triggers.

    Crisis iff EITHER `vix_value > vix_threshold` OR
    `realised_vol_pct > vol_percentile_threshold`. None-valued inputs are
    treated as "trigger did not fire" (NOT as crisis). If both inputs are
    None the result is `is_crisis=False` with empty reasons; caller should
    treat that as a data-availability problem upstream.

    Parameters
    ----------
    vix_value:
        Latest VIX spot reading. None when VIX data not available.
    realised_vol_pct:
        Latest realised-vol percentile (0-1) -- typically from
        `compute_vol_percentile_current` on a per-instrument vol history.
        None when insufficient history.
    vix_threshold, vol_percentile_threshold:
        Override the V3.8 defaults (30.0 and 0.90 respectively) if needed.

    Returns:
    -------
    `RegimeResult` with `is_crisis`, which trigger(s) fired, and
    human-readable reasons.
    """
    vix_triggered = vix_value is not None and vix_value > vix_threshold
    vol_triggered = realised_vol_pct is not None and realised_vol_pct > vol_percentile_threshold

    reasons: list[str] = []
    if vix_triggered:
        reasons.append(f"VIX {vix_value:.2f} > {vix_threshold:.2f}")
    if vol_triggered:
        reasons.append(
            f"realised-vol percentile {realised_vol_pct:.3f} > {vol_percentile_threshold:.3f}"
        )

    return RegimeResult(
        is_crisis=vix_triggered or vol_triggered,
        vix_value=vix_value,
        vix_threshold=vix_threshold,
        vix_triggered=vix_triggered,
        realised_vol_pct=realised_vol_pct,
        realised_vol_pct_threshold=vol_percentile_threshold,
        realised_vol_triggered=vol_triggered,
        reasons=tuple(reasons),
    )
