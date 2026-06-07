"""Drawdown throttle primitive (V3.8 §4.3 + §4.6.3 graded de-risking ladder).

Per `directives/Objective Reframe 2026-05-23.md` §4.3:

    Trigger: rolling portfolio DD from 60d peak > 8%.
    Action: halve all per-strategy `f` (Kelly fraction) -- sizing -> 0.125x Kelly.
    Reset: when portfolio recovers to within -4% of 60d peak.
    Rationale: sequence-risk control. 10 consecutive losses at 2% each compounds
    to -18% DD if unthrottled. Throttling at -8% caps the consecutive-loss tail
    at -12-14%.

This module implements the -8% trigger / -4% reset half-Kelly step. The
deeper -12% pre-emptive flatten lives in `leverage_envelope.py` (§4.6 C4);
the -15% PRM kill switch is V3.7 inherited. Together they form the four-step
ladder from §4.6.3.

Key design choice: hysteresis
=============================

The trigger and reset thresholds are DELIBERATELY different (-8% vs -4%).
Using a single threshold would cause flapping when DD oscillates near -8%:
the throttle would engage, equity would recover slightly back above -8%,
the throttle would disengage, equity falls again, repeat. Flapping is bad
because (a) each transition incurs sizing churn / re-rebalancing costs and
(b) the Kelly halving's protective effect is undermined when it's only
applied for a few bars at a time.

The 4-percentage-point hysteresis band (between trigger -8% and reset -4%)
means once throttled, the portfolio has to recover materially before
returning to full Kelly. This matches the V3.5 lesson: DD breakers are
failsafes, not primary controls -- once tripped, stay tripped until clear
recovery is evident.

Rolling vs cumulative DD
========================

V3.8 uses ROLLING DD from a 60-bar peak, not cumulative MaxDD from the
all-time peak. The reason: a strategy that hit -20% in 2020 then recovered
to a fresh all-time high in 2022 should not be permanently throttled by
the 2020 event. The rolling-60d-peak design reacts to RECENT losses while
forgetting old ones once 60+ bars of recovery have passed.

60-bar window matches the spec in §4.3. For non-daily bar timeframes
(H1, M5), use the framework's `BARS_PER_YEAR[tf]` to derive an equivalent
"60 trading days" window if needed -- but the V3.8 directive specifies
daily portfolio NAV bars by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# V3.8 §4.3 defaults.
DEFAULT_TRIGGER_DD: float = -0.08  # Trigger when rolling DD reaches -8%.
DEFAULT_RESET_DD: float = -0.04  # Reset when DD recovers to within -4%.
DEFAULT_PEAK_WINDOW_BARS: int = 60  # Rolling 60-bar peak.
DEFAULT_THROTTLE_MULTIPLIER: float = 0.5  # Halve Kelly when throttled.
NORMAL_MULTIPLIER: float = 1.0  # Untouched Kelly fraction.


@dataclass(frozen=True)
class DdThrottleState:
    """Stateful throttle snapshot.

    Held across bars by the caller (typically the allocator). Updated via
    `update_throttle(prev_state, current_dd)` each bar.

    Attributes:
    ----------
    multiplier:
        Current Kelly-fraction multiplier. Either `NORMAL_MULTIPLIER` (1.0)
        or `DEFAULT_THROTTLE_MULTIPLIER` (0.5) under defaults.
    current_dd:
        Latest rolling DD from 60-bar peak (non-positive float).
    triggered:
        True iff currently in the throttled state (i.e. multiplier < 1.0).
    triggered_at:
        Timestamp at which the most recent trigger fired. `None` when
        never triggered or after a reset.
    """

    multiplier: float
    current_dd: float
    triggered: bool
    triggered_at: pd.Timestamp | None


@dataclass(frozen=True)
class DdThrottlePath:
    """Result of walking a full equity path through the throttle.

    Returned by `simulate_throttle_path` for backtest / audit use. The
    `multiplier` series is what the allocator would have applied bar-by-bar
    if the throttle were live during the period covered by `equity`.

    Attributes:
    ----------
    rolling_dd:
        Per-bar DD from rolling N-bar peak.
    multiplier:
        Per-bar Kelly multiplier (1.0 or 0.5 under defaults).
    triggered:
        Per-bar boolean of throttle-engaged state.
    n_trigger_events:
        Count of distinct trigger episodes (rising edges of `triggered`).
        Useful for "how many times would this have fired in backtest".
    bars_throttled:
        Total count of bars in the throttled state.
    """

    rolling_dd: pd.Series
    multiplier: pd.Series
    triggered: pd.Series
    n_trigger_events: int
    bars_throttled: int


def compute_rolling_dd_from_peak(
    equity: pd.Series | np.ndarray | Iterable[float],
    *,
    peak_window_bars: int = DEFAULT_PEAK_WINDOW_BARS,
) -> pd.Series:
    """Per-bar drawdown from a rolling N-bar peak.

    Distinct from cumulative MaxDD (which uses the all-time peak): this
    measure forgets old peaks after `peak_window_bars`, which is what
    V3.8 §4.3 needs to react to recent losses without being permanently
    throttled by an old event.

    Equity convention: input is the portfolio NAV path (cumulative
    equity), not per-bar returns. To derive equity from returns use
    `(1 + returns).cumprod()` first.

    Parameters
    ----------
    equity:
        Portfolio NAV path, indexed by bar timestamp.
    peak_window_bars:
        Rolling window size for the peak. Default 60 per V3.8 §4.3.

    Returns:
    -------
    Series of non-positive floats, same index as `equity`. Values are
    fractional drawdowns from the rolling peak (e.g. -0.08 = 8% below
    rolling peak).
    """
    s = pd.Series(equity)
    if len(s) == 0:
        return s.copy()
    rolling_peak = s.rolling(peak_window_bars, min_periods=1).max()
    return (s - rolling_peak) / rolling_peak


def compute_throttle_multiplier(
    current_dd: float,
    *,
    trigger_dd: float = DEFAULT_TRIGGER_DD,
    throttle_multiplier: float = DEFAULT_THROTTLE_MULTIPLIER,
) -> float:
    """Stateless multiplier lookup (no hysteresis).

    Returns the throttle multiplier given the current DD with NO memory
    of prior triggers. This is the simplest possible throttle and is
    appropriate when the caller wants point-in-time behaviour for plotting,
    sanity-checking, or unit testing.

    For LIVE use the stateful `update_throttle` should be preferred
    because it avoids on/off flapping when DD oscillates near the trigger
    threshold (§4.3 hysteresis design rationale).

    Parameters
    ----------
    current_dd:
        Current drawdown (non-positive float; -0.10 = 10% DD).
    trigger_dd:
        Threshold below which the multiplier engages. Default -0.08.
    throttle_multiplier:
        Multiplier applied when triggered. Default 0.5.

    Returns:
    -------
    `throttle_multiplier` if `current_dd <= trigger_dd`, else `NORMAL_MULTIPLIER`.
    """
    return throttle_multiplier if current_dd <= trigger_dd else NORMAL_MULTIPLIER


def initial_throttle_state(timestamp: pd.Timestamp | None = None) -> DdThrottleState:
    """Build the starting throttle state (un-triggered, multiplier=1.0)."""
    return DdThrottleState(
        multiplier=NORMAL_MULTIPLIER,
        current_dd=0.0,
        triggered=False,
        triggered_at=timestamp if timestamp is not None else None,
    )


def update_throttle(
    prev_state: DdThrottleState,
    current_dd: float,
    *,
    timestamp: pd.Timestamp | None = None,
    trigger_dd: float = DEFAULT_TRIGGER_DD,
    reset_dd: float = DEFAULT_RESET_DD,
    throttle_multiplier: float = DEFAULT_THROTTLE_MULTIPLIER,
) -> DdThrottleState:
    """Apply hysteresis: engage at `trigger_dd`, release at `reset_dd`.

    Transition rules:
    1. Not triggered + DD <= trigger_dd  -> ENGAGE  (multiplier -> throttle)
    2. Triggered     + DD >= reset_dd    -> RELEASE (multiplier -> normal)
    3. Otherwise: state persists.

    Parameters
    ----------
    prev_state:
        Previous-bar throttle state. Use `initial_throttle_state()` to
        seed on first call.
    current_dd:
        Current rolling DD (e.g. from `compute_rolling_dd_from_peak`).
    timestamp:
        Optional bar timestamp -- recorded on rising edge into
        `DdThrottleState.triggered_at`.
    trigger_dd, reset_dd, throttle_multiplier:
        Override defaults if needed.

    Returns:
    -------
    New `DdThrottleState`. The previous state is not mutated (the
    dataclass is frozen).
    """
    if not prev_state.triggered and current_dd <= trigger_dd:
        # Rising edge into throttled state.
        return DdThrottleState(
            multiplier=throttle_multiplier,
            current_dd=current_dd,
            triggered=True,
            triggered_at=timestamp,
        )
    if prev_state.triggered and current_dd >= reset_dd:
        # Falling edge: recovery complete.
        return DdThrottleState(
            multiplier=NORMAL_MULTIPLIER,
            current_dd=current_dd,
            triggered=False,
            triggered_at=None,
        )
    # Hold state; update only the DD reading.
    return DdThrottleState(
        multiplier=prev_state.multiplier,
        current_dd=current_dd,
        triggered=prev_state.triggered,
        triggered_at=prev_state.triggered_at,
    )


def simulate_throttle_path(
    equity: pd.Series | np.ndarray | Iterable[float],
    *,
    trigger_dd: float = DEFAULT_TRIGGER_DD,
    reset_dd: float = DEFAULT_RESET_DD,
    peak_window_bars: int = DEFAULT_PEAK_WINDOW_BARS,
    throttle_multiplier: float = DEFAULT_THROTTLE_MULTIPLIER,
) -> DdThrottlePath:
    """Walk an equity path bar-by-bar through the throttle, applying hysteresis.

    For backtest / audit use only -- LIVE callers maintain their own
    `DdThrottleState` across `update_throttle` calls bar-by-bar. This
    function is the offline equivalent that produces the full per-bar
    multiplier series the live throttle would have produced.

    Parameters
    ----------
    equity:
        Portfolio NAV path, indexed by bar timestamp.
    trigger_dd, reset_dd, peak_window_bars, throttle_multiplier:
        Override defaults if needed.

    Returns:
    -------
    `DdThrottlePath` with rolling_dd, multiplier, triggered per-bar series
    plus event-count summary fields.
    """
    s = pd.Series(equity)
    if len(s) == 0:
        empty = pd.Series([], dtype=float)
        return DdThrottlePath(
            rolling_dd=empty,
            multiplier=empty,
            triggered=pd.Series([], dtype=bool),
            n_trigger_events=0,
            bars_throttled=0,
        )

    dd = compute_rolling_dd_from_peak(s, peak_window_bars=peak_window_bars)
    multipliers: list[float] = []
    triggered_flags: list[bool] = []
    state = initial_throttle_state()
    n_trigger_events = 0
    for ts, dd_val in dd.items():
        was_triggered = state.triggered
        state = update_throttle(
            state,
            float(dd_val),
            timestamp=ts,
            trigger_dd=trigger_dd,
            reset_dd=reset_dd,
            throttle_multiplier=throttle_multiplier,
        )
        if state.triggered and not was_triggered:
            n_trigger_events += 1
        multipliers.append(state.multiplier)
        triggered_flags.append(state.triggered)

    mult_series = pd.Series(multipliers, index=s.index)
    trig_series = pd.Series(triggered_flags, index=s.index)
    return DdThrottlePath(
        rolling_dd=dd,
        multiplier=mult_series,
        triggered=trig_series,
        n_trigger_events=n_trigger_events,
        bars_throttled=int(trig_series.sum()),
    )


__all__ = [
    "DEFAULT_TRIGGER_DD",
    "DEFAULT_RESET_DD",
    "DEFAULT_PEAK_WINDOW_BARS",
    "DEFAULT_THROTTLE_MULTIPLIER",
    "NORMAL_MULTIPLIER",
    "DdThrottleState",
    "DdThrottlePath",
    "compute_rolling_dd_from_peak",
    "compute_throttle_multiplier",
    "initial_throttle_state",
    "update_throttle",
    "simulate_throttle_path",
]
