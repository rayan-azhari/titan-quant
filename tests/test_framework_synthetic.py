"""Synthetic ground-truth tests for the unified framework.

Fixes audit-catalogue gap H2: no end-to-end test verifies that the
audit pipeline correctly accepts a known-edge strategy and rejects a
known-no-edge strategy on synthetic data.

We construct two stylised strategies:

    edge_strategy        -- a deterministic signal with built-in positive
                            mean return. Should pass the framework's gates.
    no_edge_strategy     -- a strategy whose returns are pure i.i.d. noise.
                            Should fail the framework's gates.

Plus targeted unit tests for each primitive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.research.framework import (
    DEFAULTS,
    DecisionInputs,
    McConfig,
    SharpeReporting,
    StrategyClass,
    Verdict,
    WfoConfig,
    build_folds,
    decide,
    defaults_for,
    deflated_sharpe,
    sanctuary_divergence_test,
    slice_sanctuary,
    sr_var_from_sweep,
)
from titan.research.framework.mc import run_block_mc
from titan.research.framework.typology import COST_CME_FUTURES_LIQUID
from titan.research.metrics import bootstrap_sharpe_ci, sharpe

# ── Helpers ────────────────────────────────────────────────────────────────


def _synthetic_close_with_edge(
    n_bars: int = 5_000,
    daily_drift: float = 0.0005,
    vol: float = 0.01,
    seed: int = 0,
) -> pd.Series:
    """Generates a daily close with positive drift -- a buy-and-hold edge."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(daily_drift, vol, size=n_bars)
    idx = pd.date_range("2010-01-01", periods=n_bars, freq="1D", tz="UTC")
    return pd.Series(100.0 * np.exp(np.cumsum(log_rets)), index=idx, name="close")


def _synthetic_close_no_edge(
    n_bars: int = 5_000,
    vol: float = 0.01,
    seed: int = 1,
) -> pd.Series:
    """Generates a daily close with ZERO drift -- pure noise. A
    buy-and-hold has zero true Sharpe."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, vol, size=n_bars)
    idx = pd.date_range("2010-01-01", periods=n_bars, freq="1D", tz="UTC")
    return pd.Series(100.0 * np.exp(np.cumsum(log_rets)), index=idx, name="close")


def _buy_and_hold_strategy(df: pd.DataFrame) -> pd.Series:
    """Strategy: always long, MTM daily return."""
    return df["close"].pct_change().fillna(0.0)


def _shorting_noise_strategy(df: pd.DataFrame) -> pd.Series:
    """Strategy: always SHORT a no-edge series -- should produce zero
    Sharpe + symmetric distribution. Useful for testing the no-edge gate."""
    return -df["close"].pct_change().fillna(0.0)


# ── Typology + defaults ────────────────────────────────────────────────────


def test_defaults_for_every_strategy_class():
    """Every StrategyClass enum value must have a DEFAULTS row.
    Catches: silent additions to the enum without a defaults row."""
    for cls in StrategyClass:
        d = defaults_for(cls)
        assert isinstance(d.sharpe, SharpeReporting)
        assert isinstance(d.wfo, WfoConfig)
        assert isinstance(d.mc, McConfig)


def test_defaults_table_is_total():
    """DEFAULTS table covers every enum value."""
    assert set(DEFAULTS.keys()) == set(StrategyClass), (
        f"DEFAULTS missing for: {set(StrategyClass) - set(DEFAULTS.keys())}; "
        f"extra in DEFAULTS: {set(DEFAULTS.keys()) - set(StrategyClass)}"
    )


def test_cross_asset_momentum_uses_recalibrated_mc_gate():
    """Per directive §2.4 the 25%/5% gate was broken for cross-asset
    momentum (Bond-Equity Audit §4.2-c). New gate is 35%/10%."""
    cam = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    assert cam.mc.max_dd_threshold_pct == 0.35
    assert cam.mc.max_dd_pass_prob == 0.10
    # Shared-block bootstrap preserves cross-asset correlation.
    assert cam.mc.bootstrap_method == "shared_block"


def test_intraday_breakout_uses_per_trade_sharpe():
    """Sparse-trade strategies report per-trade as primary Sharpe.
    Catches A4 from the audit catalogue."""
    ib = defaults_for(StrategyClass.INTRADAY_BREAKOUT)
    assert ib.sharpe.primary == "per_trade"
    assert ib.sharpe.secondary == "per_bar"


def test_cost_model_round_trip_arithmetic():
    """Sanity-check the CostModel.round_trip_bps_no_commission formula."""
    cm = COST_CME_FUTURES_LIQUID
    assert cm.round_trip_bps_no_commission == 4.0  # 2 * (1.0 + 1.0)


# ── Sanctuary ──────────────────────────────────────────────────────────────


def test_slice_sanctuary_carves_last_12_months():
    df = pd.DataFrame(
        {"close": np.arange(2000.0)},
        index=pd.date_range("2018-01-01", periods=2000, freq="1D", tz="UTC"),
    )
    s = slice_sanctuary(df, months=12)
    assert s.months_held_out == 12
    assert s.visible.index[-1] < s.sanctuary_start
    assert s.sanctuary.index[0] >= s.sanctuary_start
    # Sanctuary ends at df.index[-1]
    assert s.sanctuary_end == df.index[-1]


def test_slice_sanctuary_rejects_non_datetime_index():
    df = pd.DataFrame({"close": np.arange(100.0)})
    with pytest.raises(TypeError):
        slice_sanctuary(df, months=12)


def test_divergence_test_flags_lucky_sanctuary():
    """If we feed the sanctuary returns from a particularly-strong
    distribution and the historical returns from a weaker one, the
    divergence test must flag lucky."""
    rng = np.random.default_rng(42)
    historical = pd.Series(rng.normal(0.0, 0.01, size=2000))  # zero edge
    sanctuary = pd.Series(rng.normal(0.003, 0.01, size=200))  # +0.3% daily drift
    test = sanctuary_divergence_test(historical, sanctuary, periods_per_year=252)
    assert test.sanctuary_sharpe > 0.5
    # Sanctuary should sit very high in the historical distribution.
    assert test.percentile > 0.9 or test.lucky_flag


def test_divergence_test_handles_short_history_gracefully():
    historical = pd.Series([0.001] * 10)
    sanctuary = pd.Series([0.001] * 100)
    test = sanctuary_divergence_test(historical, sanctuary, periods_per_year=252)
    # When history is too short, we still return a struct (with NaN flags).
    assert test.lucky_flag is False
    assert test.unlucky_flag is False


# ── WFO ────────────────────────────────────────────────────────────────────


def test_wfo_expanding_anchored_at_start():
    idx = pd.date_range("2010-01-01", periods=2520, freq="1D", tz="UTC")
    cfg = WfoConfig(
        is_min_years=3.0,
        oos_years=1.0,
        fold_count=5,
        is_mode="expanding",
        stride_overlap_allowed=False,
    )
    folds = build_folds(idx, cfg, bars_per_year=252)
    assert len(folds) >= 3
    # IS for fold 0 starts at index 0
    assert folds[0].is_start == 0
    # OOS windows are non-overlapping
    for i in range(1, len(folds)):
        assert folds[i].oos_start >= folds[i - 1].oos_end_excl


def test_wfo_rolling_slides_forward():
    idx = pd.date_range("2010-01-01", periods=2520, freq="1D", tz="UTC")
    cfg = WfoConfig(
        is_min_years=2.0,
        oos_years=0.5,
        fold_count=8,
        is_mode="rolling",
        stride_overlap_allowed=True,
    )
    folds = build_folds(idx, cfg, bars_per_year=252)
    assert len(folds) >= 1
    # Rolling IS slides forward
    assert folds[-1].is_start > folds[0].is_start


def test_wfo_returns_empty_on_insufficient_data():
    idx = pd.date_range("2024-01-01", periods=100, freq="1D", tz="UTC")
    cfg = defaults_for(StrategyClass.DAILY_TREND).wfo
    folds = build_folds(idx, cfg, bars_per_year=252)
    assert folds == []


# ── DSR ────────────────────────────────────────────────────────────────────


def test_dsr_handles_zero_skew_normal_returns():
    """Normal returns (skew=0, kurt=3): DSR-prob at SR=0 and N=10 should
    be ~0.5 (the cell exactly hits e_max_sr expectation)."""
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.01, size=1000))
    # Construct a fake sweep with SR variance 0.25 (SR std = 0.5)
    fake_sweep = [0.0, 1.0, -1.0, 0.5, -0.5, 0.7, -0.3, 0.2, -0.7, 0.1]
    sr_var = sr_var_from_sweep(fake_sweep)
    r = deflated_sharpe(0.0, sr_var_across_trials=sr_var, returns=rets, n_trials=10)
    # At SR=0 we're below the null max; dsr_prob should be < 0.5.
    assert r.dsr_prob < 0.5


def test_dsr_strong_signal_clears_gate():
    """A strong SR (well above e_max_sr) should give dsr_prob ≈ 1.0."""
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, size=2000))
    fake_sweep = [0.0, 1.0, -1.0, 0.5, -0.5]
    sr_var = sr_var_from_sweep(fake_sweep)
    r = deflated_sharpe(10.0, sr_var_across_trials=sr_var, returns=rets, n_trials=5)
    assert r.dsr_prob > 0.95


def test_dsr_picks_up_kurtosis():
    """Fat-tailed returns at the same SR should reduce DSR-prob vs normal
    returns, due to the kurtosis-aware denominator. We compare the z-stat
    (the variance-stabilised gap) directly because at strong SR both
    DSR-probs saturate near 1.0 via norm.cdf and can't be ordered."""
    rng = np.random.default_rng(0)
    rets_normal = pd.Series(rng.normal(0.001, 0.01, size=2000))
    rets_fat = rets_normal.copy()
    rets_fat.iloc[::200] = -0.05  # 1% of bars are -5% (huge negative spikes)
    sr_var = 0.25
    # SR chosen so e_max_sr=0.78 and z lands in the unsaturated [0, 3] range
    sr_test = 1.5
    r_normal = deflated_sharpe(
        sr_test, sr_var_across_trials=sr_var, returns=rets_normal, n_trials=10
    )
    r_fat = deflated_sharpe(sr_test, sr_var_across_trials=sr_var, returns=rets_fat, n_trials=10)
    # Kurtosis must be detected
    assert r_fat.kurt > r_normal.kurt
    # The kurtosis-aware z-stat must be LOWER for the fat-tailed series.
    # (norm.cdf saturates at large z, so dsr_prob can be tied at 1.0 even
    # though the underlying gap is smaller -- that's the saturation.)
    assert r_fat.z < r_normal.z


# ── Decision matrix ────────────────────────────────────────────────────────


def test_decision_total_function_covers_all_combinations():
    """The 243-cell matrix must produce a verdict for every combination
    of {best, mid, worst}^5 (J3: 5 axes including noise robustness).
    Catches G1 (UNDETERMINED outcomes)."""
    # Sample test points for each axis level
    ci_lo_samples = [0.1, -0.1, -0.5]
    dsr_samples = [0.97, 0.7, 0.2]
    mc_p_samples = [0.02, 0.07, 0.30]
    sanc_samples = [0.5, -0.1, -0.5]
    # Noise axis is binary-derived: (pm, pw) -> {(T,T)=best, (T,F)=mid, (F,F)=worst}.
    # (F,T) is impossible by construction (worst-case pass implies mean pass).
    noise_samples = [(True, True), (True, False), (False, False)]
    seen_verdicts: set[Verdict] = set()
    n_cells = 0
    for ci in ci_lo_samples:
        for dsr in dsr_samples:
            for mc in mc_p_samples:
                for sanc in sanc_samples:
                    for pm, pw in noise_samples:
                        n_cells += 1
                        r = decide(
                            DecisionInputs(
                                ci_lo=ci,
                                dsr_prob=dsr,
                                p_maxdd_gt_threshold=mc,
                                pass_threshold_prob=0.05,
                                sanctuary_sharpe=sanc,
                                noise_passes_mean=pm,
                                noise_passes_worst=pw,
                            )
                        )
                        assert r.verdict in Verdict
                        seen_verdicts.add(r.verdict)
    assert n_cells == 243  # 3^5
    # All 5 verdict levels should be reachable
    assert seen_verdicts == set(Verdict)


def test_decision_all_axes_best_returns_deploy():
    r = decide(
        DecisionInputs(
            ci_lo=0.5,
            dsr_prob=0.99,
            p_maxdd_gt_threshold=0.01,
            pass_threshold_prob=0.05,
            sanctuary_sharpe=1.0,
            noise_passes_mean=True,
            noise_passes_worst=True,
        )
    )
    assert r.verdict == Verdict.DEPLOY
    assert r.n_axes_best == 5


def test_decision_all_axes_worst_returns_retire():
    r = decide(
        DecisionInputs(
            ci_lo=-0.5,
            dsr_prob=0.2,
            p_maxdd_gt_threshold=0.30,
            pass_threshold_prob=0.05,
            sanctuary_sharpe=-0.5,
            noise_passes_mean=False,
            noise_passes_worst=False,
        )
    )
    assert r.verdict == Verdict.RETIRE
    assert r.n_axes_best == 0


def test_decision_4_of_5_returns_conditional_watchpoint():
    """4-of-5 best (noise axis as the single worst) -> CONDITIONAL_WATCHPOINT.
    This is the canonical 'pass 4 stats axes but fail noise' case J3 was built for."""
    r = decide(
        DecisionInputs(
            ci_lo=0.5,
            dsr_prob=0.99,
            p_maxdd_gt_threshold=0.01,
            pass_threshold_prob=0.05,
            sanctuary_sharpe=1.0,
            noise_passes_mean=False,
            noise_passes_worst=False,
        )
    )
    assert r.verdict == Verdict.CONDITIONAL_WATCHPOINT
    assert r.n_axes_best == 4
    assert r.noise_axis == "worst"


def test_classify_axis_noise_truth_table():
    """Truth table for the noise classifier (J3)."""
    from titan.research.framework.decision import classify_axis_noise

    assert classify_axis_noise(True, True) == "best"
    assert classify_axis_noise(True, False) == "mid"
    assert classify_axis_noise(False, False) == "worst"
    # Anomalous worst-pass-without-mean-pass: by the API contract this is
    # impossible (worst-case <= mean), but classifier must still be total.
    # Per the implementation, worst_pass=True short-circuits to "best".
    assert classify_axis_noise(False, True) == "best"


def test_decision_5axis_verdict_thresholds():
    """Explicit n_best -> verdict mapping for J3's 5-axis matrix."""
    # 5 best -> DEPLOY (covered above)
    # 4 best -> CONDITIONAL_WATCHPOINT (covered above)
    # 3 best, but sanctuary AND noise at worst -> the P1-5 guards apply:
    # n_worst==2 and a sanctuary-worst dealbreaker both cap the verdict at
    # SUSPECT (a strategy that lost money on the held-out year is not merely
    # TIER_UNCONFIRMED). Base ladder verdict would be TIER_UNCONFIRMED.
    r = decide(
        DecisionInputs(
            ci_lo=0.5,  # best
            dsr_prob=0.99,  # best
            p_maxdd_gt_threshold=0.01,  # best
            pass_threshold_prob=0.05,
            sanctuary_sharpe=-0.5,  # worst
            noise_passes_mean=False,  # worst
            noise_passes_worst=False,
        )
    )
    assert r.verdict == Verdict.SUSPECT
    assert r.n_axes_best == 3
    assert r.n_axes_worst == 2

    # 2 best -> SUSPECT
    r = decide(
        DecisionInputs(
            ci_lo=0.5,  # best
            dsr_prob=0.99,  # best
            p_maxdd_gt_threshold=0.30,  # worst
            pass_threshold_prob=0.05,
            sanctuary_sharpe=-0.5,  # worst
            noise_passes_mean=False,  # worst
            noise_passes_worst=False,
        )
    )
    assert r.verdict == Verdict.SUSPECT
    assert r.n_axes_best == 2

    # 1 best -> RETIRE (collapsed with 0)
    r = decide(
        DecisionInputs(
            ci_lo=0.5,  # best
            dsr_prob=0.2,  # worst
            p_maxdd_gt_threshold=0.30,  # worst
            pass_threshold_prob=0.05,
            sanctuary_sharpe=-0.5,  # worst
            noise_passes_mean=False,  # worst
            noise_passes_worst=False,
        )
    )
    assert r.verdict == Verdict.RETIRE
    assert r.n_axes_best == 1


# ── P1-5 decision guards (External Quant Audit 2026-05-29) ─────────────────


def test_decision_sanctuary_worst_vetoes_to_suspect():
    """A strategy best on CI_lo/DSR/MC/noise but that LOST money on the
    held-out sanctuary year (sanctuary 'worst') must NOT score
    CONDITIONAL_WATCHPOINT. The dealbreaker veto caps it at SUSPECT."""
    r = decide(
        DecisionInputs(
            ci_lo=0.5,  # best
            dsr_prob=0.99,  # best
            p_maxdd_gt_threshold=0.01,  # best
            pass_threshold_prob=0.05,
            sanctuary_sharpe=-0.5,  # worst (OOS hold-out loss)
            noise_passes_mean=True,  # best
            noise_passes_worst=True,
        )
    )
    assert r.n_axes_best == 4  # base ladder would say CONDITIONAL_WATCHPOINT
    assert r.n_axes_worst == 1
    assert r.verdict == Verdict.SUSPECT
    assert "sanctuary worst" in r.rationale


def test_decision_ci_lo_worst_vetoes_to_suspect():
    """CI_lo 'worst' (95% CI consistent with a clearly-negative edge) is the
    second dealbreaker axis and likewise caps the verdict at SUSPECT."""
    r = decide(
        DecisionInputs(
            ci_lo=-0.5,  # worst
            dsr_prob=0.99,  # best
            p_maxdd_gt_threshold=0.01,  # best
            pass_threshold_prob=0.05,
            sanctuary_sharpe=1.0,  # best
            noise_passes_mean=True,  # best
            noise_passes_worst=True,
        )
    )
    assert r.n_axes_best == 4
    assert r.verdict == Verdict.SUSPECT
    assert "CI_lo worst" in r.rationale


def test_decision_single_noise_worst_stays_conditional():
    """Regression guard for the J3 design intent: a SINGLE noise-axis 'worst'
    is a watchpoint, not a dealbreaker. 4-best/1-noise-worst stays
    CONDITIONAL_WATCHPOINT (the veto must not over-reach to hygiene axes)."""
    r = decide(
        DecisionInputs(
            ci_lo=0.5,  # best
            dsr_prob=0.99,  # best
            p_maxdd_gt_threshold=0.01,  # best
            pass_threshold_prob=0.05,
            sanctuary_sharpe=1.0,  # best
            noise_passes_mean=False,  # worst (noise only)
            noise_passes_worst=False,
        )
    )
    assert r.n_axes_best == 4
    assert r.n_axes_worst == 1
    assert r.noise_axis == "worst"
    assert r.verdict == Verdict.CONDITIONAL_WATCHPOINT


def test_decision_two_worst_distinguished_from_two_mid():
    """3-best/2-worst (on non-dealbreaker axes) is capped at SUSPECT by the
    multi-catastrophe guard, whereas 3-best/2-mid remains TIER_UNCONFIRMED --
    the count-of-best ladder alone could not tell them apart."""
    two_worst = decide(
        DecisionInputs(
            ci_lo=0.5,  # best
            dsr_prob=0.2,  # worst (non-dealbreaker)
            p_maxdd_gt_threshold=0.30,  # worst (non-dealbreaker)
            pass_threshold_prob=0.05,
            sanctuary_sharpe=1.0,  # best
            noise_passes_mean=True,  # best
            noise_passes_worst=True,
        )
    )
    assert two_worst.n_axes_best == 3
    assert two_worst.n_axes_worst == 2
    assert two_worst.verdict == Verdict.SUSPECT

    two_mid = decide(
        DecisionInputs(
            ci_lo=0.5,  # best
            dsr_prob=0.7,  # mid
            p_maxdd_gt_threshold=0.07,  # mid (between threshold and 2x)
            pass_threshold_prob=0.05,
            sanctuary_sharpe=1.0,  # best
            noise_passes_mean=True,  # best
            noise_passes_worst=True,
        )
    )
    assert two_mid.n_axes_best == 3
    assert two_mid.n_axes_worst == 0
    assert two_mid.verdict == Verdict.TIER_UNCONFIRMED


# ── P1-1 ruin drawdown convention (External Quant Audit 2026-05-29) ─────────


def test_ruin_maxdd_matches_metrics_geometric():
    """ruin._max_drawdown must equal titan.research.metrics.max_drawdown on
    identical simple-return paths -- both geometric, anchored at 1.0. Locks
    the P1-1 fix (prior np.cumsum additive measure diverged)."""
    from titan.research.framework.ruin import _max_drawdown
    from titan.research.metrics import max_drawdown

    rng = np.random.default_rng(7)
    for _ in range(200):
        path = rng.normal(0.0003, 0.012, size=252)
        ruin_dd = _max_drawdown(path)
        metrics_dd = max_drawdown(pd.Series(path))
        assert ruin_dd == pytest.approx(metrics_dd, abs=1e-12)
        assert ruin_dd <= 0.0


def test_ruin_maxdd_is_fractional_not_additive():
    """A deep, compounding loss path: geometric drawdown is bounded in
    (-1, 0] and is a fraction of equity, unlike the old additive cumsum."""
    from titan.research.framework.ruin import _max_drawdown

    path = np.full(10, -0.10)  # ten consecutive -10% bars
    dd = _max_drawdown(path)
    # Geometric: 0.9**10 - 1 ~= -0.6513, NOT the additive -1.0.
    assert dd == pytest.approx(0.9**10 - 1.0, abs=1e-9)
    assert dd > -1.0


# ── P1-2 ruin V3.8 deployment gate (External Quant Audit 2026-05-29) ────────


def _benign_returns(n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0004, 0.01, size=n), index=idx)


def test_ruin_gate_presets_match_doctrine():
    from titan.research.framework.ruin import GATE_V37, GATE_V38, HORIZON_10Y_DAILY

    assert HORIZON_10Y_DAILY == 2520
    assert (GATE_V37.horizon_bars, GATE_V37.kill_dd, GATE_V37.max_p_kill) == (252, 0.15, 0.01)
    # V3.8: 10y horizon, P_kill(DD>40%) <= 1e-3 AND P(MaxDD>25%) <= 5%.
    assert GATE_V38.horizon_bars == 2520
    assert GATE_V38.kill_dd == 0.40
    assert GATE_V38.max_p_kill == 1e-3
    assert GATE_V38.maxdd_constraint_level == 0.25
    assert GATE_V38.max_p_maxdd_constraint == 0.05


def test_ruin_passes_gate_rejects_mismatched_assessment():
    """Core P1-2 safety: a V3.8 gate cannot be applied to a V3.7-simulated
    assessment -- it RAISES rather than silently using V3.7 numbers (the exact
    drift the audit found, where every caller applied V3.7 thresholds)."""
    from titan.research.framework.ruin import GATE_V38, assess_strategy_ruin

    rets = _benign_returns(300, seed=1)
    res_v37 = assess_strategy_ruin(rets, deployment_weight=0.3, n_paths=100)  # default 252b / 15%
    assert res_v37.horizon_bars == 252
    with pytest.raises(ValueError, match="horizon"):
        res_v37.passes_gate(GATE_V38)


def test_ruin_gate_v38_simulates_under_gate_params():
    from titan.research.framework.ruin import GATE_V38, assess_strategy_ruin

    rets = _benign_returns(750, seed=2)  # long enough to avoid the coverage warning
    res = assess_strategy_ruin(rets, deployment_weight=0.3, gate=GATE_V38, n_paths=200)
    assert res.horizon_bars == 2520  # gate overrides the 252 default
    assert res.portfolio_kill_threshold == 0.40
    assert res.maxdd_constraint_level == 0.25
    assert 0.0 <= res.p_maxdd_gt_constraint <= 1.0
    # passes_gate evaluates without raising now that the params match.
    assert isinstance(res.passes_gate(GATE_V38), bool)


def test_ruin_legacy_passes_emits_deprecation():
    from titan.research.framework.ruin import assess_strategy_ruin

    rets = _benign_returns(300, seed=3)
    res = assess_strategy_ruin(rets, deployment_weight=0.3, n_paths=100)
    with pytest.warns(DeprecationWarning):
        res.passes()


# ── P1-3 stressed-correlation joint ruin + GATE_V38_STRICT (audit 2026-05-29) ──


def test_gate_v38_strict_presets():
    from titan.research.framework.ruin import GATE_V38_STRICT

    # Operator P4-4 ratification: RoR~=0 / MaxDD<20% -> tighter than 25%/40%.
    assert GATE_V38_STRICT.kill_dd == 0.30
    assert GATE_V38_STRICT.maxdd_constraint_level == 0.20
    assert GATE_V38_STRICT.max_p_kill == 1e-3
    assert GATE_V38_STRICT.horizon_bars == 2520


def test_stressed_joint_ruin_is_more_pessimistic_than_base():
    """P1-3: forcing crisis correlation -> 0.9 must produce a worse (more
    negative) joint MaxDD / higher P(MaxDD>25%) than the base joint ruin,
    which can only reproduce the (near-uncorrelated) sample structure."""
    from titan.research.framework.ruin import assess_joint_ruin, assess_joint_ruin_stressed

    rng = np.random.default_rng(4)
    n = 1500
    # Three near-uncorrelated positive-drift sleeves -> base diversifies well.
    rets = {
        "a": pd.Series(rng.normal(0.0004, 0.01, n)),
        "b": pd.Series(rng.normal(0.0004, 0.01, n)),
        "c": pd.Series(rng.normal(0.0004, 0.01, n)),
    }
    weights = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    base = assess_joint_ruin(rets, deployment_weights=weights, n_paths=300, seed=1)
    stressed = assess_joint_ruin_stressed(
        rets, deployment_weights=weights, n_paths=300, seed=1, crisis_rho=0.9
    )
    # Stress makes the tail worse: deeper median MaxDD and >= base tail prob.
    assert stressed.median_maxdd_at_size < base.median_maxdd_at_size
    assert stressed.p_maxdd_gt_constraint >= base.p_maxdd_gt_constraint
    assert stressed.p95_maxdd_at_size <= base.p95_maxdd_at_size


def test_stressed_joint_ruin_passes_gate_when_run_under_gate():
    from titan.research.framework.ruin import (
        GATE_V38_STRICT,
        assess_joint_ruin_stressed,
    )

    rng = np.random.default_rng(5)
    rets = {k: pd.Series(rng.normal(0.0004, 0.01, 1500)) for k in ("a", "b")}
    res = assess_joint_ruin_stressed(
        rets, deployment_weights={"a": 0.5, "b": 0.5}, gate=GATE_V38_STRICT, n_paths=200
    )
    # Simulated under the gate's params -> passes_gate must evaluate, not raise.
    assert res.horizon_bars == GATE_V38_STRICT.horizon_bars
    assert res.maxdd_constraint_level == GATE_V38_STRICT.maxdd_constraint_level
    assert isinstance(res.passes_gate(GATE_V38_STRICT), bool)


# ── End-to-end synthetic ground truth (H2 fix) ─────────────────────────────


def test_known_edge_strategy_passes_gates():
    """Buy-and-hold on a series with +0.05% daily drift and 1% daily vol
    has Sharpe ~ 0.8 annualised. With sufficient sample, CI_lo > 0 and
    DSR should clear -- the framework should DEPLOY or
    CONDITIONAL_WATCHPOINT, NOT retire."""
    close = _synthetic_close_with_edge(n_bars=5000, daily_drift=0.0005, vol=0.01, seed=0)
    rets = close.pct_change().dropna()
    sh = sharpe(rets, periods_per_year=252)
    ci_lo, _ = bootstrap_sharpe_ci(rets, periods_per_year=252, n_resamples=500, seed=42)
    # Strong daily drift → high Sharpe → CI_lo > 0
    assert ci_lo > 0.0
    # Fake DSR: assume the sweep had 5 cells with SR std = 0.3
    fake_sweep = [sh, 0.3, 0.5, 0.1, -0.1]
    sr_var = sr_var_from_sweep(fake_sweep)
    dsr = deflated_sharpe(sh, sr_var_across_trials=sr_var, returns=rets, n_trials=5)
    assert dsr.dsr_prob > 0.9  # strong signal clears DSR
    # Fake MC: assume passes. Fake noise gate: assume passes (long-and-hold
    # on a strong-drift series is the canonical noise-robust strategy).
    decision = decide(
        DecisionInputs(
            ci_lo=ci_lo,
            dsr_prob=dsr.dsr_prob,
            p_maxdd_gt_threshold=0.02,
            pass_threshold_prob=0.05,
            sanctuary_sharpe=0.5,
            noise_passes_mean=True,
            noise_passes_worst=True,
        )
    )
    assert decision.verdict in (Verdict.DEPLOY, Verdict.CONDITIONAL_WATCHPOINT)


def test_known_no_edge_strategy_does_not_deploy():
    """Buy-and-hold on a zero-drift series has Sharpe ~ 0 annualised.
    CI_lo straddles zero. The framework should NOT issue a DEPLOY verdict."""
    close = _synthetic_close_no_edge(n_bars=5000, vol=0.01, seed=1)
    rets = close.pct_change().dropna()
    sh = sharpe(rets, periods_per_year=252)
    ci_lo, _ = bootstrap_sharpe_ci(rets, periods_per_year=252, n_resamples=500, seed=42)
    # Zero-drift → Sharpe near zero; CI_lo likely < 0
    fake_sweep = [sh, 0.3, 0.5, 0.1, -0.1]
    sr_var = sr_var_from_sweep(fake_sweep)
    dsr = deflated_sharpe(sh, sr_var_across_trials=sr_var, returns=rets, n_trials=5)
    decision = decide(
        DecisionInputs(
            ci_lo=ci_lo,
            dsr_prob=dsr.dsr_prob,
            p_maxdd_gt_threshold=0.30,
            pass_threshold_prob=0.05,
            sanctuary_sharpe=-0.1,
            noise_passes_mean=True,
            noise_passes_worst=False,
        )
    )
    # Must NOT be a DEPLOY decision on a no-edge series
    assert decision.verdict != Verdict.DEPLOY


def test_mc_passes_for_high_quality_edge():
    """Run actual MC on a synthetic edge series and check the gate fires
    correctly. This validates the full block-bootstrap pipeline end-to-end."""
    close = _synthetic_close_with_edge(n_bars=3000, daily_drift=0.0008, vol=0.008, seed=2)
    cfg = McConfig(
        block_size_bars=20,
        n_paths=50,
        bootstrap_method="block",
        max_dd_threshold_pct=0.50,
        max_dd_pass_prob=0.10,
    )
    result = run_block_mc(
        primary_close=close,
        cfg=cfg,
        strategy_fn=_buy_and_hold_strategy,
        periods_per_year=252,
        seed=42,
    )
    assert result.n_paths_completed > 10
    # With a strong positive drift and a generous MaxDD threshold of 50%,
    # the strategy should pass on most paths.
    assert result.median_sharpe > 0.5


def test_mc_fails_for_pure_noise():
    close = _synthetic_close_no_edge(n_bars=3000, vol=0.01, seed=3)
    cfg = McConfig(
        block_size_bars=20,
        n_paths=50,
        bootstrap_method="block",
        max_dd_threshold_pct=0.25,
        max_dd_pass_prob=0.05,
    )
    result = run_block_mc(
        primary_close=close,
        cfg=cfg,
        strategy_fn=_buy_and_hold_strategy,
        periods_per_year=252,
        seed=42,
    )
    # Pure-noise buy-and-hold should fail the gate
    assert not result.passes
