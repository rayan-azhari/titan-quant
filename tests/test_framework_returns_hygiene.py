"""Unit tests for the P1-22 return/volatility hygiene primitives
(`titan/research/framework/returns_hygiene.py`).

Tests cover:
    1. mask_nonpositive_prices models anomalies (CL settle) as NaN.
    2. returns_from_prices drops ffill-before-returns (no phantom 0%).
    3. realised_vol / n_vol_observations exclude pure-missing but keep cash 0.0.
    4. mark_pure_missing partitions cash-vs-missing.
    5. The phantom-zero-deflates-vol demonstration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.research.framework.returns_hygiene import (
    mark_pure_missing,
    mask_nonpositive_prices,
    n_vol_observations,
    realised_vol,
    returns_from_prices,
)


def test_mask_nonpositive_prices():
    p = pd.Series([100.0, -37.0, 0.0, 102.0])
    m = mask_nonpositive_prices(p)
    assert m.iloc[0] == 100.0
    assert np.isnan(m.iloc[1])  # CL negative settle -> missing
    assert np.isnan(m.iloc[2])  # zero -> missing
    assert m.iloc[3] == 102.0


def test_returns_no_ffill_yields_nan_at_gaps():
    p = pd.Series([100.0, 102.0, np.nan, 103.0])
    r = returns_from_prices(p, kind="simple")
    assert r.iloc[1] == pytest.approx(0.02)
    assert np.isnan(r.iloc[2])  # gap -> NaN, NOT a phantom 0%
    assert np.isnan(r.iloc[3])  # return out of the gap also undefined


def test_returns_log_matches_and_validates():
    p = pd.Series([100.0, 110.0, 121.0])
    r = returns_from_prices(p, kind="log")
    assert r.iloc[1] == pytest.approx(np.log(1.1))
    assert r.iloc[2] == pytest.approx(np.log(1.1))
    with pytest.raises(ValueError, match="kind"):
        returns_from_prices(p, kind="arithmetic")


def test_cl_settle_excluded_not_phantom():
    # The 2020-04-20 CL -$37 settle must NOT produce a huge return or a flat 0%.
    p = pd.Series([50.0, 51.0, -37.0, 52.0, 53.0])
    r = returns_from_prices(mask_nonpositive_prices(p), kind="simple")
    # Returns into AND out of the anomaly are undefined (excluded).
    assert np.isnan(r.iloc[2]) and np.isnan(r.iloc[3])
    finite = r.dropna()
    # No catastrophic spurious return survives.
    assert finite.abs().max() < 0.5


def test_realised_vol_keeps_cash_drops_missing():
    cash = pd.Series([0.01, 0.0, -0.01, 0.02, -0.015])  # a genuine cash 0.0
    missing = pd.Series([0.01, np.nan, -0.01, 0.02, -0.015])  # pure-missing
    assert n_vol_observations(cash) == 5
    assert n_vol_observations(missing) == 4
    # Dropping the (NaN) missing day vs keeping the (0.0) cash day gives a
    # different denominator -> different annualised vol.
    assert realised_vol(cash, 252) != realised_vol(missing, 252)


def test_realised_vol_too_short_is_nan():
    assert np.isnan(realised_vol(pd.Series([np.nan, 0.01]), 252))
    assert np.isnan(realised_vol(pd.Series([0.01]), 252))


def test_mark_pure_missing_partitions():
    rets = pd.Series([0.01, 0.0, -0.01, 0.0])
    valid = pd.Series([True, True, False, True])  # idx2 had no real print
    out = mark_pure_missing(rets, valid)
    assert out.iloc[0] == 0.01
    assert out.iloc[1] == 0.0  # genuine cash day kept
    assert np.isnan(out.iloc[2])  # pure-missing -> excluded
    assert out.iloc[3] == 0.0


def test_phantom_zeros_deflate_vol():
    # A price gap, handled two ways.
    p = pd.Series([100.0, 102.0, 101.0, np.nan, 103.0, 104.0, 102.0, 105.0])
    hygienic = returns_from_prices(p, kind="simple")  # NaN around the gap
    phantom = p.ffill().pct_change(fill_method=None)  # ffill -> 0% phantom day
    # The hygienic series has FEWER observations in the vol denominator...
    assert n_vol_observations(hygienic) < n_vol_observations(phantom)
    # ...and the phantom-filled vol differs (it is contaminated by fake zeros).
    assert realised_vol(hygienic, 252) != realised_vol(phantom, 252)
