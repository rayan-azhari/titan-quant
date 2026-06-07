"""Unit tests for the crisis-regime detector
(`titan/strategies/regime_filter/regime.py`).

Covers V3.8 §4.6 C3:
    1. compute_realised_vol_annualised: math identity + ppy scaling.
    2. compute_vol_percentile_current: rank percentile + None-on-insufficient.
    3. is_crisis_regime four-quadrant OR logic
       (neither triggers / VIX triggers / vol triggers / both trigger).
    4. None-tolerant inputs: missing VIX / missing vol percentile.
    5. Custom thresholds respected.
    6. Public-API contract via both regime_filter and framework __init__.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from titan.strategies.regime_filter.regime import (
    DEFAULT_VIX_THRESHOLD,
    DEFAULT_VOL_PERCENTILE_THRESHOLD,
    RegimeResult,
    compute_realised_vol_annualised,
    compute_vol_percentile_current,
    is_crisis_regime,
)

# ── compute_realised_vol_annualised ────────────────────────────────────────


def test_realised_vol_constant_return_is_zero_vol():
    """Constant returns have zero rolling std -> zero realised vol."""
    s = pd.Series([0.001] * 100)
    vol = compute_realised_vol_annualised(s, window_bars=5)
    # First 4 bars are NaN; rest are 0.
    assert vol.iloc[:4].isna().all()
    assert vol.iloc[4:].sum() == 0.0


def test_realised_vol_scales_with_sqrt_periods():
    """std=0.01 daily -> ann vol = 0.01 * sqrt(252) ~= 15.87%. Use a long
    history + large window so the rolling-std estimate converges to truth."""
    rng = np.random.default_rng(42)
    # 2000 IID returns with known std -- enough that the mean of rolling
    # 60-bar stds tracks the population std closely.
    s = pd.Series(rng.normal(loc=0.0, scale=0.01, size=2000))
    vol = compute_realised_vol_annualised(s, window_bars=60, periods_per_year=252)
    expected = 0.01 * math.sqrt(252)
    assert vol.dropna().mean() == pytest.approx(expected, rel=0.05)


def test_realised_vol_explicit_ppy_required_no_hidden_default():
    """L60 guardrail: caller must pass periods_per_year explicitly. Default
    of 252 is provided but caller is expected to override for non-daily bars."""
    s = pd.Series([0.01, -0.01, 0.005, -0.005, 0.01, -0.01])
    # H1 bar: ppy = 252 * 6.5 = 1638 ish. Use 1638 explicitly.
    vol_d = compute_realised_vol_annualised(s, window_bars=3, periods_per_year=252)
    vol_h = compute_realised_vol_annualised(s, window_bars=3, periods_per_year=1638)
    # H1 vol should be sqrt(1638/252) = ~2.55x bigger than the daily.
    ratio = vol_h.dropna().mean() / vol_d.dropna().mean()
    assert ratio == pytest.approx(math.sqrt(1638 / 252), rel=1e-6)


def test_realised_vol_window_warmup_is_nan():
    s = pd.Series(range(10), dtype=float).diff().fillna(0.0)
    vol = compute_realised_vol_annualised(s, window_bars=5)
    assert vol.iloc[:4].isna().all()
    assert not vol.iloc[4:].isna().any()


# ── compute_vol_percentile_current ─────────────────────────────────────────


def test_vol_percentile_latest_is_max_returns_near_one():
    """If the latest value is the maximum, percentile is ~1.0."""
    history = pd.Series(list(range(60)) + [1000.0])
    p = compute_vol_percentile_current(history, lookback_bars=70)
    # Latest = 1000, strictly greater than 60 prior values -> rank ~= 60/61.
    assert p == pytest.approx(60 / 61, abs=0.01)


def test_vol_percentile_latest_is_min_returns_near_zero():
    """If the latest value is the minimum, percentile is ~0."""
    history = pd.Series([100.0] * 60 + [-1.0])
    p = compute_vol_percentile_current(history, lookback_bars=70)
    # Latest = -1, strictly less than 60 prior values -> rank = 0 / 61.
    assert p == pytest.approx(0.0, abs=0.02)


def test_vol_percentile_latest_is_median_returns_near_half():
    """If the latest is the median, percentile is ~0.5."""
    # 60 evenly-spaced values 1..60 + latest=30.5 (true median).
    history = pd.Series(list(range(1, 61)) + [30.5])
    p = compute_vol_percentile_current(history, lookback_bars=70)
    assert p == pytest.approx(0.5, abs=0.05)


def test_vol_percentile_insufficient_history_returns_none():
    """<60 non-NaN observations -> None."""
    history = pd.Series([1.0, 2.0, 3.0])
    assert compute_vol_percentile_current(history, lookback_bars=10) is None


def test_vol_percentile_empty_returns_none():
    assert compute_vol_percentile_current(pd.Series([], dtype=float)) is None


def test_vol_percentile_drops_nans_before_count():
    """NaNs in the history shouldn't count toward the 60-observation threshold."""
    history = pd.Series([np.nan] * 100 + list(range(60)) + [30.0])
    p = compute_vol_percentile_current(history)
    assert p is not None
    assert 0.4 < p < 0.6  # median-ish


def test_vol_percentile_respects_lookback_window():
    """A recent regime shift should change the percentile when an old chunk
    of values slides out of the lookback window."""
    # 100 highs at 100, then 60 lows at 10, then current value = 50.
    history = pd.Series([100.0] * 100 + [10.0] * 60 + [50.0])
    # lookback=61 = the 60 prior lows + the current value only.
    # Current value 50 strictly greater than all 60 lows -> percentile ~= 60/61.
    p_short = compute_vol_percentile_current(history, lookback_bars=61)
    assert p_short is not None
    assert p_short == pytest.approx(60 / 61, abs=0.01)
    # lookback=200 spans the highs and lows. Current value 50 is middle of
    # the distribution -- 60 values (10s) below, 100 values (100s) above.
    p_long = compute_vol_percentile_current(history, lookback_bars=200)
    assert p_long is not None
    # 60/161 = 0.373 -> percentile in [0.2, 0.5].
    assert 0.2 < p_long < 0.5


# ── is_crisis_regime ───────────────────────────────────────────────────────


def test_neither_trigger_returns_not_crisis():
    res = is_crisis_regime(vix_value=15.0, realised_vol_pct=0.50)
    assert not res.is_crisis
    assert not res.vix_triggered
    assert not res.realised_vol_triggered
    assert res.reasons == ()


def test_vix_alone_triggers_crisis():
    res = is_crisis_regime(vix_value=35.0, realised_vol_pct=0.50)
    assert res.is_crisis
    assert res.vix_triggered
    assert not res.realised_vol_triggered
    assert len(res.reasons) == 1
    assert "VIX" in res.reasons[0]


def test_vol_percentile_alone_triggers_crisis():
    res = is_crisis_regime(vix_value=15.0, realised_vol_pct=0.95)
    assert res.is_crisis
    assert not res.vix_triggered
    assert res.realised_vol_triggered
    assert len(res.reasons) == 1
    assert "realised-vol percentile" in res.reasons[0]


def test_both_triggers_fire_simultaneously():
    res = is_crisis_regime(vix_value=40.0, realised_vol_pct=0.99)
    assert res.is_crisis
    assert res.vix_triggered
    assert res.realised_vol_triggered
    assert len(res.reasons) == 2


def test_vix_exactly_at_threshold_does_not_trigger():
    """Spec is 'VIX > 30', strict inequality. VIX = 30 stays normal regime."""
    res = is_crisis_regime(vix_value=DEFAULT_VIX_THRESHOLD, realised_vol_pct=0.50)
    assert not res.vix_triggered


def test_vol_percentile_exactly_at_threshold_does_not_trigger():
    """Spec is 'vol > 90th percentile', strict inequality."""
    res = is_crisis_regime(vix_value=15.0, realised_vol_pct=DEFAULT_VOL_PERCENTILE_THRESHOLD)
    assert not res.realised_vol_triggered


def test_none_vix_treated_as_not_triggered():
    res = is_crisis_regime(vix_value=None, realised_vol_pct=0.50)
    assert not res.vix_triggered
    assert not res.is_crisis


def test_none_vol_percentile_treated_as_not_triggered():
    res = is_crisis_regime(vix_value=15.0, realised_vol_pct=None)
    assert not res.realised_vol_triggered
    assert not res.is_crisis


def test_both_inputs_none_returns_not_crisis_with_empty_reasons():
    """Data-availability fail-safe: missing both inputs -> not crisis.
    Caller is expected to handle the data gap upstream."""
    res = is_crisis_regime(vix_value=None, realised_vol_pct=None)
    assert not res.is_crisis
    assert res.reasons == ()


def test_custom_thresholds_respected():
    """Tighter thresholds engage earlier; looser later."""
    # VIX 22 below default 30 -> normal. With tighter 20 threshold -> crisis.
    res_default = is_crisis_regime(vix_value=22.0, realised_vol_pct=0.50)
    assert not res_default.vix_triggered
    res_strict = is_crisis_regime(vix_value=22.0, realised_vol_pct=0.50, vix_threshold=20.0)
    assert res_strict.vix_triggered


def test_regime_result_is_frozen():
    res = is_crisis_regime(vix_value=15.0, realised_vol_pct=0.50)
    assert isinstance(res, RegimeResult)
    with pytest.raises(Exception):
        res.is_crisis = True  # type: ignore[misc]


# ── Public-API contract ────────────────────────────────────────────────────


def test_regime_symbols_exported_from_regime_filter_init():
    from titan.strategies.regime_filter import (  # noqa: F401
        DEFAULT_VIX_THRESHOLD,
        DEFAULT_VOL_PERCENTILE_THRESHOLD,
        DEFAULT_VOL_PERCENTILE_WINDOW_BARS,
        DEFAULT_VOL_WINDOW_BARS,
        RegimeResult,
        compute_realised_vol_annualised,
        compute_vol_percentile_current,
        is_crisis_regime,
    )


def test_regime_symbols_exported_from_framework():
    """V3.8 primitive re-export symmetry with calmar / dd_throttle /
    leverage_envelope: regime_filter is also reachable from
    titan.research.framework even though its physical path is in
    titan.strategies/."""
    from titan.research.framework import (  # noqa: F401
        DEFAULT_VIX_THRESHOLD,
        DEFAULT_VOL_PERCENTILE_THRESHOLD,
        RegimeResult,
        compute_realised_vol_annualised,
        compute_vol_percentile_current,
        is_crisis_regime,
    )
