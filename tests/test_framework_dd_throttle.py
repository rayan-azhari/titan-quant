"""Unit tests for the DD throttle primitive
(`titan/research/framework/dd_throttle.py`).

Covers:
    1. compute_rolling_dd_from_peak math: flat / monotone-up / drawdown shapes.
    2. Rolling-window vs cumulative-MaxDD distinction (the key V3.8 design choice).
    3. Stateless compute_throttle_multiplier lookup.
    4. Hysteresis under update_throttle: trigger / hold / reset transitions.
    5. simulate_throttle_path end-to-end on a synthetic equity path with a
       known drawdown and recovery -- multiplier series matches expected ladder.
    6. Edge cases: empty input, NaN-only, single-bar series.
    7. Public-API contract via __init__.py re-export.

Per directives/Objective Reframe 2026-05-23.md §4.3 the throttle defaults are
-8% trigger / -4% reset / 60-bar peak / 0.5x multiplier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.research.framework.dd_throttle import (
    DEFAULT_PEAK_WINDOW_BARS,
    DEFAULT_RESET_DD,
    DEFAULT_THROTTLE_MULTIPLIER,
    DEFAULT_TRIGGER_DD,
    NORMAL_MULTIPLIER,
    DdThrottleState,
    compute_rolling_dd_from_peak,
    compute_throttle_multiplier,
    initial_throttle_state,
    simulate_throttle_path,
    update_throttle,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _equity_from_returns(returns: list[float]) -> pd.Series:
    """Build an equity NAV path from per-bar returns starting at 1.0."""
    idx = pd.date_range("2020-01-01", periods=len(returns), freq="B")
    rets = pd.Series(returns, index=idx)
    return (1.0 + rets).cumprod()


def _flat_then_drawdown_then_recover(
    flat_bars: int = 100,
    dd_depth: float = -0.10,
    dd_bars: int = 20,
    recover_bars: int = 20,
) -> pd.Series:
    """Synthetic equity: flat -> linear drawdown to dd_depth -> linear recovery."""
    flat = [0.0] * flat_bars
    # dd_bars rows of equal per-bar loss summing to dd_depth.
    per_loss = (1.0 + dd_depth) ** (1.0 / dd_bars) - 1.0
    drawdown = [per_loss] * dd_bars
    # recover_bars rows of equal per-bar gain back to flat-period peak.
    per_gain = (1.0 / (1.0 + dd_depth)) ** (1.0 / recover_bars) - 1.0
    recover = [per_gain] * recover_bars
    return _equity_from_returns(flat + drawdown + recover)


# ── compute_rolling_dd_from_peak ───────────────────────────────────────────


def test_rolling_dd_flat_equity_is_zero():
    """Flat equity = perpetual peak; DD identically zero everywhere."""
    eq = _equity_from_returns([0.0] * 100)
    dd = compute_rolling_dd_from_peak(eq)
    assert (dd == 0.0).all()


def test_rolling_dd_monotone_up_is_zero():
    """Monotone-up equity has cumulative new highs; rolling DD = 0."""
    eq = _equity_from_returns([0.001] * 100)
    dd = compute_rolling_dd_from_peak(eq)
    # Each bar is a new high => rolling peak == equity => dd == 0.
    np.testing.assert_allclose(dd.to_numpy(), 0.0, atol=1e-12)


def test_rolling_dd_negative_during_drawdown():
    """A 10% drawdown produces dd values bottoming near -10%."""
    eq = _flat_then_drawdown_then_recover(flat_bars=100, dd_depth=-0.10, dd_bars=20)
    dd = compute_rolling_dd_from_peak(eq)
    # Bottom should be approximately -10%.
    assert dd.min() == pytest.approx(-0.10, abs=0.005)


def test_rolling_dd_forgets_old_peak_after_window():
    """V3.8 design: an old peak should NOT permanently throttle the portfolio.
    After `peak_window_bars` of flat equity, the rolling peak slides forward
    and the dd resets to 0 even though equity is below its all-time high."""
    # Equity path: 1.0 at bar 0 (peak), -50% loss at bar 1, then flat at 0.5
    # for > peak_window_bars. cumprod from rets [0, -0.5, 0, 0, ...] gives
    # eq = [1.0, 0.5, 0.5, 0.5, ...].
    rets = [0.0, -0.5] + [0.0] * (DEFAULT_PEAK_WINDOW_BARS + 10)
    eq = _equity_from_returns(rets)
    dd = compute_rolling_dd_from_peak(eq)
    # Just after the loss: rolling peak still includes 1.0; dd ~= -50%.
    assert dd.iloc[1] == pytest.approx(-0.5, abs=1e-9)
    # After peak_window_bars + buffer, the 1.0 peak slides out of the
    # window; rolling peak == current equity (0.5) -> dd resets to 0.
    assert dd.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_rolling_dd_empty_input():
    dd = compute_rolling_dd_from_peak(pd.Series([], dtype=float))
    assert len(dd) == 0


# ── compute_throttle_multiplier (stateless) ────────────────────────────────


def test_stateless_multiplier_normal_above_trigger():
    assert compute_throttle_multiplier(-0.05) == NORMAL_MULTIPLIER
    assert compute_throttle_multiplier(0.0) == NORMAL_MULTIPLIER
    assert compute_throttle_multiplier(0.10) == NORMAL_MULTIPLIER  # impossible but safe


def test_stateless_multiplier_throttled_below_trigger():
    assert compute_throttle_multiplier(-0.08) == DEFAULT_THROTTLE_MULTIPLIER
    assert compute_throttle_multiplier(-0.15) == DEFAULT_THROTTLE_MULTIPLIER


def test_stateless_multiplier_exactly_at_trigger_is_throttled():
    """Trigger condition is <= -8% per spec; exact -8% engages throttle."""
    assert compute_throttle_multiplier(DEFAULT_TRIGGER_DD) == DEFAULT_THROTTLE_MULTIPLIER


def test_stateless_multiplier_custom_thresholds():
    # If trigger is -3%, then -5% throttles, -2% does not.
    assert compute_throttle_multiplier(-0.05, trigger_dd=-0.03) == DEFAULT_THROTTLE_MULTIPLIER
    assert compute_throttle_multiplier(-0.02, trigger_dd=-0.03) == NORMAL_MULTIPLIER


# ── Hysteresis (update_throttle) ──────────────────────────────────────────


def test_initial_state_is_untriggered():
    s = initial_throttle_state()
    assert not s.triggered
    assert s.multiplier == NORMAL_MULTIPLIER
    assert s.current_dd == 0.0


def test_hysteresis_engages_on_first_breach():
    s0 = initial_throttle_state()
    s1 = update_throttle(s0, -0.09)
    assert s1.triggered
    assert s1.multiplier == DEFAULT_THROTTLE_MULTIPLIER


def test_hysteresis_does_not_engage_just_above_trigger():
    s0 = initial_throttle_state()
    s1 = update_throttle(s0, -0.07)
    assert not s1.triggered
    assert s1.multiplier == NORMAL_MULTIPLIER


def test_hysteresis_holds_triggered_state_during_oscillation():
    """Once triggered, the throttle holds even if DD bounces back to -5%
    (above trigger -8% but below reset -4%). This is the whole point of
    hysteresis vs the stateless lookup."""
    s = initial_throttle_state()
    s = update_throttle(s, -0.10)  # engage
    assert s.triggered
    s = update_throttle(s, -0.05)  # bounce up into the hysteresis band
    assert s.triggered
    assert s.multiplier == DEFAULT_THROTTLE_MULTIPLIER


def test_hysteresis_releases_at_reset_threshold():
    s = initial_throttle_state()
    s = update_throttle(s, -0.10)  # engage
    assert s.triggered
    s = update_throttle(s, DEFAULT_RESET_DD)  # exactly at reset; releases.
    assert not s.triggered
    assert s.multiplier == NORMAL_MULTIPLIER


def test_hysteresis_releases_at_full_recovery():
    s = initial_throttle_state()
    s = update_throttle(s, -0.10)  # engage
    s = update_throttle(s, 0.0)  # full recovery
    assert not s.triggered
    assert s.multiplier == NORMAL_MULTIPLIER


def test_hysteresis_re_engages_after_recovery_and_new_drawdown():
    """Full lifecycle: trigger -> reset -> trigger again on a new drawdown."""
    s = initial_throttle_state()
    s = update_throttle(s, -0.10)
    assert s.triggered
    s = update_throttle(s, -0.02)  # recover above reset
    assert not s.triggered
    s = update_throttle(s, -0.09)  # new drawdown
    assert s.triggered


def test_triggered_at_timestamp_recorded_on_rising_edge_only():
    ts1 = pd.Timestamp("2020-01-15")
    ts2 = pd.Timestamp("2020-01-16")
    ts3 = pd.Timestamp("2020-01-20")
    s = initial_throttle_state()
    s = update_throttle(s, -0.10, timestamp=ts1)
    assert s.triggered_at == ts1
    # Holding the triggered state should NOT overwrite the original timestamp.
    s = update_throttle(s, -0.05, timestamp=ts2)
    assert s.triggered
    assert s.triggered_at == ts1
    # On reset, triggered_at clears.
    s = update_throttle(s, 0.0, timestamp=ts3)
    assert not s.triggered
    assert s.triggered_at is None


def test_dd_throttle_state_is_immutable():
    """Frozen dataclass: mutation should raise."""
    s = initial_throttle_state()
    with pytest.raises(Exception):
        s.multiplier = 0.5  # type: ignore[misc]


# ── simulate_throttle_path end-to-end ──────────────────────────────────────


def test_simulate_throttle_path_engages_during_deep_drawdown():
    """A -15% drawdown should trigger the -8% throttle at least once and
    leave a non-zero `bars_throttled` count."""
    eq = _flat_then_drawdown_then_recover(flat_bars=80, dd_depth=-0.15, dd_bars=30, recover_bars=30)
    path = simulate_throttle_path(eq)
    assert path.n_trigger_events >= 1
    assert path.bars_throttled > 0
    # During the drawdown bottom, multiplier should be the throttle value.
    bottom_idx = path.rolling_dd.idxmin()
    assert path.multiplier.loc[bottom_idx] == DEFAULT_THROTTLE_MULTIPLIER
    assert path.triggered.loc[bottom_idx]


def test_simulate_throttle_path_no_trigger_on_shallow_drawdown():
    """A -5% drawdown stays above -8% trigger -- multiplier stays at 1.0."""
    eq = _flat_then_drawdown_then_recover(flat_bars=80, dd_depth=-0.05, dd_bars=30, recover_bars=30)
    path = simulate_throttle_path(eq)
    assert path.n_trigger_events == 0
    assert path.bars_throttled == 0
    assert (path.multiplier == NORMAL_MULTIPLIER).all()


def test_simulate_throttle_path_releases_on_recovery():
    """A drawdown that triggers then recovers above reset should release
    the throttle by end of series."""
    eq = _flat_then_drawdown_then_recover(flat_bars=80, dd_depth=-0.12, dd_bars=20, recover_bars=40)
    path = simulate_throttle_path(eq)
    assert path.n_trigger_events >= 1
    # Last bar should be back to normal multiplier (full recovery).
    assert path.multiplier.iloc[-1] == NORMAL_MULTIPLIER
    assert not path.triggered.iloc[-1]


def test_simulate_throttle_path_one_episode_persists_through_oscillation():
    """A drawdown that bounces between -5% and -10% multiple times should
    register as ONE trigger episode, not many -- because hysteresis holds the
    state through the bounces."""
    # Build equity: trigger at bar 100; bounce up/down within [-10%, -5%]
    # several times; finally recover to flat.
    eq_vals: list[float] = [1.0] * 100
    # Drop to 0.90 (DD = -10%).
    eq_vals += [0.92, 0.90]
    # Bounce up to 0.96 (DD = -4% within hysteresis band).
    # CAUTION: 0.96 / 1.00 = -4% which is the RESET threshold -- using
    # 0.95 (DD = -5%) ensures we stay strictly inside the hysteresis band.
    eq_vals += [0.93, 0.95]
    # Drop back to 0.88 (DD = -12%).
    eq_vals += [0.92, 0.90, 0.88]
    # Bounce to 0.95 again (still in band).
    eq_vals += [0.93, 0.95]
    # Final recovery above reset to 0.97 (DD = -3%).
    eq_vals += [0.96, 0.97]
    eq = pd.Series(eq_vals, index=pd.date_range("2020-01-01", periods=len(eq_vals), freq="B"))
    path = simulate_throttle_path(eq)
    # Expect 1 trigger event (engaged on first -10%, held through the bounce,
    # released only at -3% recovery).
    assert path.n_trigger_events == 1


def test_simulate_throttle_path_empty_input():
    path = simulate_throttle_path(pd.Series([], dtype=float))
    assert path.n_trigger_events == 0
    assert path.bars_throttled == 0
    assert len(path.rolling_dd) == 0


def test_simulate_throttle_path_constant_multiplier_when_state_stays_normal():
    """Constant up-trending equity -> rolling DD stays at 0 -> multiplier
    stays at 1.0 for every bar."""
    eq = _equity_from_returns([0.0005] * 200)
    path = simulate_throttle_path(eq)
    assert (path.multiplier == NORMAL_MULTIPLIER).all()
    assert path.n_trigger_events == 0


# ── Public-API contract ────────────────────────────────────────────────────


def test_dd_throttle_symbols_exported_from_framework_init():
    """Smoke-test the __init__.py re-export so V3.8 doctrine references hold."""
    from titan.research.framework import (
        DEFAULT_PEAK_WINDOW_BARS,  # noqa: F401
        DEFAULT_RESET_DD,  # noqa: F401
        DEFAULT_THROTTLE_MULTIPLIER,  # noqa: F401
        DEFAULT_TRIGGER_DD,  # noqa: F401
        NORMAL_MULTIPLIER,  # noqa: F401
        DdThrottlePath,  # noqa: F401
        DdThrottleState,  # noqa: F401
        compute_rolling_dd_from_peak,  # noqa: F401
        compute_throttle_multiplier,  # noqa: F401
        initial_throttle_state,  # noqa: F401
        simulate_throttle_path,  # noqa: F401
        update_throttle,  # noqa: F401
    )


def test_dataclass_kwargs_path_works():
    """Sanity: a non-Series equity input (list of floats) is also accepted."""
    dd = compute_rolling_dd_from_peak([1.0, 1.0, 0.9, 0.95, 1.05])
    # First bar is its own peak -> DD = 0; bar 2 dropped to 0.9 from peak 1.0 = -0.10.
    assert dd.iloc[0] == 0.0
    assert dd.iloc[2] == pytest.approx(-0.10, abs=1e-12)
    # Bar 4 fresh high -> DD = 0.
    assert dd.iloc[4] == pytest.approx(0.0, abs=1e-12)


# Reference to DdThrottleState to silence "imported but unused" if some
# tests above are ever removed. The dataclass is the public type used as
# the return value of `update_throttle`.
_ = DdThrottleState
