"""Synthetic ground-truth tests for the IC-analysis module.

We build a panel with a *known* relationship between factor and forward return:

    edge factor      factor[t] == next-bar return + small noise. Cross-sectional
                     IC at h=1 must be strongly positive, decay as the horizon
                     grows, and the top quantile must out-return the bottom.
    no-edge factor   factor independent of returns. IC must be ~0 and the
                     quantile spread insignificant.

Plus targeted unit tests: forward_returns alignment, the metrics.sharpe routing
identity for the annualised IC-IR, and error/edge handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.research import metrics
from titan.research.framework import (
    IcSummary,
    QuantileResult,
    cross_sectional_ic,
    forward_returns,
    ic_decay,
    quantile_returns,
    rolling_ic,
    summarise_ic,
)

T, K = 300, 30  # 300 dates, 30-symbol universe
DATES = pd.date_range("2022-01-03", periods=T, freq="B")
SYMS = [f"S{i:02d}" for i in range(K)]


def _panels(noise: float = 0.005, seed: int = 0):
    """Build (prices, edge_factor, noedge_factor) panels with a known edge.

    Returns are i.i.d. normal. ``edge_factor[t]`` equals the *next-bar* log
    return plus noise, so it predicts ``forward_returns(prices, 1)`` by
    construction. ``noedge_factor`` is independent noise.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, size=(T, K))  # rets[t] is the return INTO bar t
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))

    # forward_returns(prices, 1)[t] = log(prices[t+1]/prices[t]) = rets[t+1]
    fwd1 = np.full((T, K), np.nan)
    fwd1[:-1] = rets[1:]
    edge = fwd1 + rng.normal(0.0, noise, size=(T, K))
    noedge = rng.normal(0.0, 0.01, size=(T, K))

    px_df = pd.DataFrame(prices, index=DATES, columns=SYMS)
    edge_df = pd.DataFrame(edge, index=DATES, columns=SYMS)
    noedge_df = pd.DataFrame(noedge, index=DATES, columns=SYMS)
    return px_df, edge_df, noedge_df


# ── forward_returns ───────────────────────────────────────────────────────────


def test_forward_returns_alignment_and_shape():
    px = pd.Series([100.0, 110.0, 121.0, 133.1], index=range(4))
    fr = forward_returns(px, 1, log=True)
    # r_t = log(px_{t+1}/px_t); each step is +10% => log(1.1)
    assert fr.iloc[0] == pytest.approx(np.log(1.1))
    assert fr.iloc[1] == pytest.approx(np.log(1.1))
    assert np.isnan(fr.iloc[-1])  # no future price for the last bar

    simple = forward_returns(px, 1, log=False)
    assert simple.iloc[0] == pytest.approx(0.1)


def test_forward_returns_horizon_and_validation():
    px = pd.Series(np.arange(1, 11), dtype=float)
    fr2 = forward_returns(px, 2, log=False)
    assert fr2.iloc[0] == pytest.approx(3.0 / 1.0 - 1.0)
    assert np.isnan(fr2.iloc[-1]) and np.isnan(fr2.iloc[-2])
    with pytest.raises(ValueError):
        forward_returns(px, 0)


# ── cross-sectional IC: edge vs no-edge ──────────────────────────────────────


def test_cross_sectional_ic_detects_edge():
    px, edge, _ = _panels()
    fr = forward_returns(px, 1)
    ic = cross_sectional_ic(edge, fr, method="spearman")
    assert len(ic) > T - 10  # almost every date qualifies
    assert ic.mean() > 0.3  # a genuine, strong positive rank-IC


def test_cross_sectional_ic_rejects_noise():
    px, _, noedge = _panels()
    fr = forward_returns(px, 1)
    ic = cross_sectional_ic(noedge, fr, method="spearman")
    assert abs(ic.mean()) < 0.1  # no systematic relationship


def test_cross_sectional_ic_requires_cross_section():
    px, edge, _ = _panels()
    fr = forward_returns(px, 1)
    with pytest.raises(ValueError, match="cross-section"):
        cross_sectional_ic(edge.iloc[:, [0]], fr.iloc[:, [0]])


def test_cross_sectional_ic_rejects_bad_method():
    px, edge, _ = _panels()
    fr = forward_returns(px, 1)
    with pytest.raises(ValueError, match="method"):
        cross_sectional_ic(edge, fr, method="kendall")


# ── summarise_ic + the metrics.sharpe routing identity ───────────────────────


def test_summarise_ic_routes_annualisation_through_metrics_sharpe():
    px, edge, _ = _panels()
    fr = forward_returns(px, 1)
    ic = cross_sectional_ic(edge, fr)
    summ = summarise_ic(ic, periods_per_year=252, horizon=1)

    assert isinstance(summ, IcSummary)
    # The annualised IC-IR is, by construction, the Sharpe of the IC series.
    assert summ.ic_ir_annualised == pytest.approx(metrics.sharpe(ic, 252))
    # Un-annualised IR and t-stat are internally consistent.
    assert summ.t_stat == pytest.approx(summ.ic_ir * np.sqrt(summ.n_periods))
    assert summ.ic_mean > 0.0
    assert 0.0 <= summ.pct_positive <= 1.0
    assert summ.pct_positive > 0.9  # strong edge => almost all dates positive


def test_summarise_ic_empty_series():
    summ = summarise_ic(pd.Series([], dtype=float), periods_per_year=252, horizon=1)
    assert summ.n_periods == 0
    assert summ.ic_mean == 0.0 and summ.ic_ir_annualised == 0.0


# ── IC decay ──────────────────────────────────────────────────────────────────


def test_ic_decay_peaks_at_h1_and_decays():
    px, edge, _ = _panels()
    decay = ic_decay(edge, px, horizons=(1, 5, 10), periods_per_year=252)
    assert [s.horizon for s in decay] == [1, 5, 10]
    by_h = {s.horizon: s.ic_mean for s in decay}
    # Factor predicts only the next bar => IC strongest at h=1, fading after.
    assert by_h[1] > by_h[5] > 0
    assert by_h[1] > by_h[10]


def test_ic_decay_requires_periods_per_year_keyword():
    px, edge, _ = _panels()
    with pytest.raises(TypeError):
        ic_decay(edge, px, horizons=(1,))  # periods_per_year is keyword-only


# ── Quantile spread ───────────────────────────────────────────────────────────


def test_quantile_returns_cross_sectional_monotone_for_edge():
    px, edge, _ = _panels()
    qr = quantile_returns(edge, px, horizon=1, n_quantiles=5, periods_per_year=252)
    assert isinstance(qr, QuantileResult)
    assert qr.cross_sectional is True
    assert qr.n_quantiles == 5
    # Top bucket out-returns bottom, ladder is (near) monotone, spread significant.
    assert qr.spread > 0.0
    assert qr.monotonicity > 0.8
    assert qr.spread_tstat > 2.0
    assert qr.mean_returns[-1] > qr.mean_returns[0]


def test_quantile_returns_noise_has_no_spread():
    px, _, noedge = _panels()
    qr = quantile_returns(noedge, px, horizon=1, n_quantiles=5, periods_per_year=252)
    assert abs(qr.spread_tstat) < 2.0  # not significant


def test_quantile_returns_pooled_single_instrument():
    px, edge, _ = _panels()
    # One symbol -> pooled (time-series) binning path.
    qr = quantile_returns(
        edge[SYMS[0]], px[SYMS[0]], horizon=1, n_quantiles=4, periods_per_year=252
    )
    assert qr.cross_sectional is False
    assert sum(qr.counts) > 0
    assert qr.mean_returns[-1] > qr.mean_returns[0]  # edge still sorts pooled


def test_quantile_returns_validates_n_quantiles():
    px, edge, _ = _panels()
    with pytest.raises(ValueError):
        quantile_returns(edge, px, horizon=1, n_quantiles=1, periods_per_year=252)


# ── rolling_ic (single instrument) ────────────────────────────────────────────


def test_rolling_ic_single_instrument_positive_for_edge():
    px, edge, _ = _panels()
    sym = SYMS[3]
    fr = forward_returns(px[sym], 1)
    ric = rolling_ic(edge[sym], fr, window=60)
    valid = ric.dropna()
    assert len(valid) > 100
    assert valid.mean() > 0.3  # per-symbol edge shows up in rolling IC
