"""Regime-filter primitive (V3.8 §4.6 C3 crisis-regime detection).

Sits in `titan/strategies/` per `directives/Objective Reframe 2026-05-23.md`
§6.3 path spec, but is a PURE FUNCTION over (VIX value, realised vol
percentile) -- not a live strategy class. The intent is for any strategy /
risk module that needs the "are we in crisis regime?" signal to import from
here, and for the eventual PRM `on_bar` hook to call `is_crisis_regime`
once per bar to drive the §4.6 C3 leverage / heat-cap reduction.

Also re-exported from `titan.research.framework` for symmetry with the
other V3.8 primitives.
"""

from titan.strategies.regime_filter.regime import (
    DEFAULT_VIX_THRESHOLD,
    DEFAULT_VOL_PERCENTILE_THRESHOLD,
    DEFAULT_VOL_PERCENTILE_WINDOW_BARS,
    DEFAULT_VOL_WINDOW_BARS,
    RegimeResult,
    compute_realised_vol_annualised,
    compute_vol_percentile_current,
    is_crisis_regime,
)

__all__ = [
    "DEFAULT_VIX_THRESHOLD",
    "DEFAULT_VOL_PERCENTILE_THRESHOLD",
    "DEFAULT_VOL_PERCENTILE_WINDOW_BARS",
    "DEFAULT_VOL_WINDOW_BARS",
    "RegimeResult",
    "compute_realised_vol_annualised",
    "compute_vol_percentile_current",
    "is_crisis_regime",
]
