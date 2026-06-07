"""Standardised walk-forward fold construction.

Specified in directives/Methodology Audit & Unified Framework 2026-05-14.md
§2.3. Fixes gaps C1 (inconsistent WFO designs) and C6 (no standard config).

Two fold modes per strategy class:

    "expanding"  — anchored at the start, IS expands each fold, non-overlapping OOS
    "rolling"    — fixed-length IS slides forward, optional OOS overlap

All folds operate on the VISIBLE portion (post-sanctuary slice). Sanctuary
discipline is enforced via the SanctuarySlice already applied upstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd

from titan.research.framework.typology import WfoConfig


@dataclass(frozen=True)
class Fold:
    """One WFO fold. Index positions into the visible DataFrame."""

    fold_id: int
    is_start: int
    is_end_excl: int  # exclusive
    oos_start: int  # == is_end_excl
    oos_end_excl: int  # exclusive
    # Wall-clock boundaries for audit logs.
    is_start_ts: pd.Timestamp
    is_end_ts: pd.Timestamp
    oos_start_ts: pd.Timestamp
    oos_end_ts: pd.Timestamp

    @property
    def n_is_bars(self) -> int:
        return self.is_end_excl - self.is_start

    @property
    def n_oos_bars(self) -> int:
        return self.oos_end_excl - self.oos_start


def build_folds(
    visible_index: pd.DatetimeIndex,
    cfg: WfoConfig,
    *,
    bars_per_year: float,
) -> list[Fold]:
    """Construct WFO folds per the per-class config.

    Parameters
    ----------
    visible_index:
        The post-sanctuary DatetimeIndex of the data.
    cfg:
        WfoConfig from typology.defaults_for(strategy_class).wfo.
    bars_per_year:
        Conversion from `cfg.is_min_years` / `cfg.oos_years` to bar
        counts. For D-frequency strategies use 252; for H1 use 252*24;
        etc.

    Returns:
    -------
    folds : list[Fold]
        Possibly empty if the visible window is too short for the
        requested IS_min + OOS configuration.
    """
    n = len(visible_index)
    if n == 0:
        return []
    is_min_bars = int(cfg.is_min_years * bars_per_year)
    oos_bars = int(cfg.oos_years * bars_per_year)
    if is_min_bars <= 0 or oos_bars <= 0:
        return []
    if n < is_min_bars + oos_bars:
        return []

    # Auto-scale fold count to span the available history (the
    # ``auto_fold_count`` lesson — explicit class defaults can be too
    # conservative for long-history data, yielding OOS coverage that's a
    # small fraction of the visible window).
    effective_fold_count = cfg.fold_count
    if cfg.auto_fold_count:
        if cfg.is_mode == "expanding":
            # k-th fold: IS_end = is_min + k * oos. Last fold: IS_end + oos <= n
            # => k <= (n - is_min - oos) / oos = (n - is_min)/oos - 1.
            max_folds = max(1, (n - is_min_bars) // oos_bars)
        else:
            # rolling: stride = oos (no-overlap) or oos//2 (overlap).
            stride = oos_bars if not cfg.stride_overlap_allowed else max(1, oos_bars // 2)
            # k-th fold: IS_start = stride*k, OOS_end = IS_start + is_min + oos <= n
            max_folds = max(1, (n - is_min_bars - oos_bars) // stride + 1)
        # Use the larger of the configured floor + the auto-derived count,
        # capped at the configured ceiling.
        effective_fold_count = min(cfg.auto_fold_count_max, max(cfg.fold_count, int(max_folds)))

    folds: list[Fold] = []
    if cfg.is_mode == "expanding":
        # Anchored at index 0; IS expands; non-overlapping OOS.
        # For fold k: IS = [0, is_min_bars + k * oos_bars), OOS = next oos_bars.
        for k in range(effective_fold_count):
            is_end = is_min_bars + k * oos_bars
            oos_end = is_end + oos_bars
            if oos_end > n:
                break
            folds.append(
                Fold(
                    fold_id=k,
                    is_start=0,
                    is_end_excl=is_end,
                    oos_start=is_end,
                    oos_end_excl=oos_end,
                    is_start_ts=visible_index[0],
                    is_end_ts=visible_index[is_end - 1],
                    oos_start_ts=visible_index[is_end],
                    oos_end_ts=visible_index[oos_end - 1],
                )
            )
    elif cfg.is_mode == "rolling":
        # Fixed-length IS that slides forward. Stride = oos_bars; overlap
        # of OOS windows is allowed if cfg.stride_overlap_allowed.
        # For fold k: IS = [stride * k, stride * k + is_min_bars), OOS = next oos_bars.
        stride = oos_bars if not cfg.stride_overlap_allowed else max(1, oos_bars // 2)
        for k in range(effective_fold_count):
            is_start = stride * k
            is_end = is_start + is_min_bars
            oos_end = is_end + oos_bars
            if oos_end > n:
                break
            folds.append(
                Fold(
                    fold_id=k,
                    is_start=is_start,
                    is_end_excl=is_end,
                    oos_start=is_end,
                    oos_end_excl=oos_end,
                    is_start_ts=visible_index[is_start],
                    is_end_ts=visible_index[is_end - 1],
                    oos_start_ts=visible_index[is_end],
                    oos_end_ts=visible_index[oos_end - 1],
                )
            )
    else:
        raise ValueError(f"Unknown WFO mode: {cfg.is_mode!r}")
    return folds


def iter_folds(
    visible: pd.DataFrame,
    cfg: WfoConfig,
    *,
    bars_per_year: float,
) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    """Convenience generator: yields ``(fold, is_df, oos_df)`` tuples."""
    folds = build_folds(visible.index, cfg, bars_per_year=bars_per_year)
    for f in folds:
        yield (
            f,
            visible.iloc[f.is_start : f.is_end_excl],
            visible.iloc[f.oos_start : f.oos_end_excl],
        )


# --------------------------------------------------------------------------- #
# CPCV — Combinatorial Purged Cross-Validation (P1-10)                         #
# --------------------------------------------------------------------------- #
#
# Specified in directives/Audit Remediation Plan 2026-05-29.md row P1-10:
# "CPCV + PBO option ... to replace the n=1 terminal window".
#
# López de Prado, *Advances in Financial Machine Learning* (2018) ch. 7 + 12.
# The single forward-chaining WFO above produces ONE out-of-sample path. CPCV
# splits the series into N contiguous groups and holds out every combination
# of k groups as the test set -- C(N, k) splits yielding C(N-1, k-1) distinct
# backtest *paths*. That distribution of paths is what feeds the PBO estimate
# (titan.research.framework.program_ledger.probability_of_backtest_overfitting)
# and gives a far more robust read on OOS performance than a single terminal
# split.
#
# Two leakage guards (essential when labels span multiple bars):
#   * PURGE   -- drop training bars within ``purge_bars`` on EITHER side of a
#                test block (their label windows overlap the test set).
#   * EMBARGO -- additionally drop training bars within an embargo buffer
#                AFTER each test block, to kill leakage via serial correlation
#                that purging alone misses (LdP §7.4).


@dataclass(frozen=True)
class CombinatorialFold:
    """One CPCV split: train + test index positions for a single combination
    of held-out groups, after purging + embargo.

    ``train_idx`` / ``test_idx`` are integer positions into the visible array
    (0-based, sorted). ``n_purged`` is the count of non-test bars dropped from
    training by the purge + embargo guards.
    """

    fold_id: int
    test_group_ids: tuple[int, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_purged: int

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_test(self) -> int:
        return int(self.test_idx.size)


def cpcv_n_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of distinct backtest paths a CPCV scheme produces.

    Each of the ``n_groups`` groups is held out in exactly ``C(N-1, k-1)`` of
    the ``C(N, k)`` splits, so every observation is tested on that many paths.
    """
    if not 1 <= n_test_groups < n_groups:
        raise ValueError(
            f"need 1 <= n_test_groups < n_groups, got n_test_groups={n_test_groups}, "
            f"n_groups={n_groups}"
        )
    return math.comb(n_groups - 1, n_test_groups - 1)


def build_cpcv_folds(
    n_obs: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_frac: float = 0.01,
    purge_bars: int = 0,
) -> list[CombinatorialFold]:
    """Construct Combinatorial Purged Cross-Validation folds.

    Parameters
    ----------
    n_obs:
        Number of observations (bars) in the visible series.
    n_groups:
        Number of contiguous, (near-)equal-size groups N to split into.
    n_test_groups:
        Groups held out as the test set per split, k (``1 <= k < N``). The
        number of splits is ``C(N, k)``; distinct paths is ``C(N-1, k-1)``.
    embargo_frac:
        Fraction of ``n_obs`` to embargo AFTER each test block (LdP §7.4).
        ``0.01`` = 1% of the sample. Set 0 to disable.
    purge_bars:
        Bars to purge on EITHER side of each test block (set to the label
        horizon so overlapping-label training bars are dropped). Default 0.

    Returns:
    -------
    list[CombinatorialFold]
        One fold per combination, in ``itertools.combinations`` order.

    Raises:
    ------
    ValueError
        On ``n_groups < 2``, ``not (1 <= n_test_groups < n_groups)``,
        ``n_obs < n_groups``, or negative ``embargo_frac`` / ``purge_bars``.
    """
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError(
            f"need 1 <= n_test_groups < n_groups, got n_test_groups={n_test_groups}, "
            f"n_groups={n_groups}"
        )
    if n_obs < n_groups:
        raise ValueError(f"n_obs ({n_obs}) must be >= n_groups ({n_groups})")
    if embargo_frac < 0:
        raise ValueError(f"embargo_frac must be >= 0, got {embargo_frac}")
    if purge_bars < 0:
        raise ValueError(f"purge_bars must be >= 0, got {purge_bars}")

    groups = np.array_split(np.arange(n_obs), n_groups)
    embargo = math.ceil(embargo_frac * n_obs)

    folds: list[CombinatorialFold] = []
    for fold_id, combo in enumerate(combinations(range(n_groups), n_test_groups)):
        test_idx = np.concatenate([groups[g] for g in combo])
        test_set = set(test_idx.tolist())

        # Purge on both sides + embargo after, around each selected block.
        excluded: set[int] = set()
        for g in combo:
            block = groups[g]
            if block.size == 0:
                continue
            lo, hi = int(block[0]), int(block[-1])
            excluded.update(range(max(0, lo - purge_bars), lo))
            after = max(purge_bars, embargo)
            excluded.update(range(hi + 1, min(n_obs, hi + 1 + after)))

        train_idx = np.array(
            [i for i in range(n_obs) if i not in test_set and i not in excluded],
            dtype=int,
        )
        n_purged = n_obs - len(test_set) - int(train_idx.size)
        folds.append(
            CombinatorialFold(
                fold_id=fold_id,
                test_group_ids=combo,
                train_idx=train_idx,
                test_idx=np.sort(test_idx),
                n_purged=n_purged,
            )
        )
    return folds
