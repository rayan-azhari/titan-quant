"""Unit tests for the P1-10 additions:
1. CPCV (Combinatorial Purged Cross-Validation) in `framework/wfo.py`.
2. Multi-window / bootstrapped-CI sanctuary in `framework/sanctuary.py`.
3. The lucky-sanctuary verdict downgrade in `framework/decision.py`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from titan.research.framework.decision import DecisionInputs, Verdict, decide
from titan.research.framework.sanctuary import (
    multi_window_sanctuary_test,
    slice_multi_sanctuary,
)
from titan.research.framework.wfo import build_cpcv_folds, cpcv_n_paths

# ── CPCV ─────────────────────────────────────────────────────────────────


def test_cpcv_n_paths_formula():
    # Each of N groups is held out in C(N-1, k-1) of the C(N, k) splits.
    assert cpcv_n_paths(6, 2) == math.comb(5, 1) == 5
    assert cpcv_n_paths(10, 2) == math.comb(9, 1) == 9
    assert cpcv_n_paths(8, 3) == math.comb(7, 2) == 21


def test_cpcv_n_paths_validation():
    with pytest.raises(ValueError):
        cpcv_n_paths(5, 5)  # k must be < N
    with pytest.raises(ValueError):
        cpcv_n_paths(5, 0)


def test_cpcv_fold_count_and_disjointness():
    folds = build_cpcv_folds(120, n_groups=6, n_test_groups=2, embargo_frac=0.0)
    assert len(folds) == math.comb(6, 2)  # 15 splits
    for f in folds:
        train = set(f.train_idx.tolist())
        test = set(f.test_idx.tolist())
        assert train.isdisjoint(test)  # no leakage between train and test
        assert len(test) == 40  # two groups of 20
        assert train.union(test).issubset(set(range(120)))
        assert f.n_test == 40


def test_cpcv_every_observation_tested_n_paths_times():
    n_obs, n_groups, k = 120, 6, 2
    folds = build_cpcv_folds(n_obs, n_groups=n_groups, n_test_groups=k, embargo_frac=0.0)
    counts = np.zeros(n_obs, dtype=int)
    for f in folds:
        counts[f.test_idx] += 1
    # Every observation is in the test set exactly C(N-1, k-1) times.
    assert set(counts.tolist()) == {cpcv_n_paths(n_groups, k)}


def test_cpcv_embargo_purges_after_test_block():
    # k=1 -> one test group per fold; embargo drops bars right after it.
    n_obs = 120
    folds = build_cpcv_folds(n_obs, n_groups=6, n_test_groups=1, embargo_frac=0.05)
    embargo = math.ceil(0.05 * n_obs)  # 6
    # Fold testing group 0 = block [0, 19]; embargo excludes 20..25 from train.
    f0 = folds[0]
    assert f0.test_group_ids == (0,)
    train = set(f0.train_idx.tolist())
    for j in range(20, 20 + embargo):
        assert j not in train
    assert f0.n_purged == embargo


def test_cpcv_purge_drops_both_sides():
    folds = build_cpcv_folds(120, n_groups=6, n_test_groups=1, embargo_frac=0.0, purge_bars=3)
    # Group 2 = block [40, 59]; purge drops 37,38,39 (before) and 60,61,62 (after).
    f = next(fd for fd in folds if fd.test_group_ids == (2,))
    train = set(f.train_idx.tolist())
    for j in (37, 38, 39, 60, 61, 62):
        assert j not in train
    assert f.n_purged == 6


def test_cpcv_input_validation():
    with pytest.raises(ValueError, match="n_groups must be"):
        build_cpcv_folds(100, n_groups=1)
    with pytest.raises(ValueError, match="n_test_groups"):
        build_cpcv_folds(100, n_groups=4, n_test_groups=4)
    with pytest.raises(ValueError, match=">= n_groups"):
        build_cpcv_folds(3, n_groups=6)
    with pytest.raises(ValueError, match="embargo_frac"):
        build_cpcv_folds(100, embargo_frac=-0.1)
    with pytest.raises(ValueError, match="purge_bars"):
        build_cpcv_folds(100, purge_bars=-1)


# ── Multi-window sanctuary ─────────────────────────────────────────────────


def _series(values: np.ndarray, *, start: str = "2019-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_slice_multi_sanctuary_disjoint_and_recent_first():
    rng = np.random.default_rng(0)
    s = _series(rng.normal(0, 0.01, size=252 * 5))  # ~5 years business days
    slices = slice_multi_sanctuary(s, n_windows=3, months=12)
    assert len(slices) == 3
    # Most-recent first: start timestamps strictly decreasing.
    starts = [sl.sanctuary_start for sl in slices]
    assert starts[0] > starts[1] > starts[2]
    # Each window's visible ends strictly before its sanctuary start.
    for sl in slices:
        if not sl.visible.empty:
            assert sl.visible.index[-1] < sl.sanctuary_start


def test_multi_window_result_shape_and_ci():
    rng = np.random.default_rng(1)
    s = _series(rng.normal(0.0, 0.01, size=252 * 5))
    res = multi_window_sanctuary_test(s, periods_per_year=252, n_windows=3, n_resamples=200)
    assert res.n_windows == 3
    assert len(res.per_window) == 3
    assert len(res.window_sharpes) == 3
    assert res.sharpe_ci_lo <= res.sharpe_ci_hi
    # Pure noise: not a majority-lucky/unlucky verdict.
    assert res.lucky_flag is False
    assert res.unlucky_flag is False


def test_multi_window_flags_lucky_recent_spike():
    # Flat history, strongly positive most-recent year -> that window lands in
    # the top 5% of historical rolling windows -> lucky.
    rng = np.random.default_rng(2)
    n = 252 * 4
    vals = rng.normal(0.0, 0.01, size=n)
    vals[-252:] += 0.01  # ~+1%/day drift in the final year
    s = _series(vals)
    res = multi_window_sanctuary_test(s, periods_per_year=252, n_windows=1, n_resamples=200)
    assert res.n_lucky == 1
    assert res.lucky_flag is True


# ── decide() lucky downgrade ───────────────────────────────────────────────


def _deploy_inputs() -> DecisionInputs:
    return DecisionInputs(
        ci_lo=0.5,
        dsr_prob=0.99,
        p_maxdd_gt_threshold=0.01,
        pass_threshold_prob=0.05,
        sanctuary_sharpe=0.5,
        noise_passes_mean=True,
        noise_passes_worst=True,
    )


def _retire_inputs() -> DecisionInputs:
    return DecisionInputs(
        ci_lo=-0.5,
        dsr_prob=0.10,
        p_maxdd_gt_threshold=0.50,
        pass_threshold_prob=0.05,
        sanctuary_sharpe=-0.5,
        noise_passes_mean=False,
        noise_passes_worst=False,
    )


def test_decide_default_unchanged_by_lucky_false():
    res = decide(_deploy_inputs())
    assert res.verdict is Verdict.DEPLOY
    assert res.lucky_downgrade_applied is False
    # Backward compatibility: explicit lucky_flag=False is identical.
    assert decide(_deploy_inputs(), lucky_flag=False).verdict is Verdict.DEPLOY


def test_decide_lucky_downgrades_one_level():
    res = decide(_deploy_inputs(), lucky_flag=True)
    assert res.verdict is Verdict.CONDITIONAL_WATCHPOINT
    assert res.lucky_downgrade_applied is True
    assert "LUCKY-SANCTUARY" in res.rationale


def test_decide_lucky_saturates_at_retire():
    res = decide(_retire_inputs(), lucky_flag=True)
    assert res.verdict is Verdict.RETIRE
    # No change possible -> not flagged as applied.
    assert res.lucky_downgrade_applied is False
