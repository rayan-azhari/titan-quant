"""Standardised block-bootstrap Monte Carlo.

Specified in directives/Methodology Audit & Unified Framework 2026-05-14.md
§2.4. Fixes gaps F1 (block-size sensitivity), F2 (broken threshold),
F3 (method ambiguity), F4 (correlation preservation), F6 (no class
defaults).

Three bootstrap methods:

    "block"        — Resample (block-size) chunks from a single return
                     series. Preserves serial autocorrelation up to
                     block boundaries.
    "shared_block" — Two or more series sampled at the SAME block
                     indices. Preserves cross-asset correlation.
    "stationary"   — Politis & Romano (1994) stationary block bootstrap
                     with GEOMETRIC (random) block lengths + circular wrap.
                     Better-preserves clustered drawdowns than fixed blocks;
                     `block_size_bars` is the MEAN block length. Extras follow
                     the same resampled indices (like shared_block).

All methods rebuild synthetic price paths via cumulative product of
resampled log returns, then re-run the strategy on each path. The
strategy's MaxDD distribution + Sharpe distribution are reported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from titan.research.framework.typology import McConfig
from titan.research.metrics import max_drawdown, sharpe

# Number of worker processes for MC parallelism. Capped at CPU count and at
# 8 to avoid OS-level oversubscription on shared dev machines.
DEFAULT_MC_WORKERS = max(1, min(8, (os.cpu_count() or 4)))


@dataclass(frozen=True)
class McResult:
    """Aggregate stats from a Monte Carlo run."""

    n_paths_completed: int
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    median_maxdd: float
    p_maxdd_gt_threshold: float
    threshold_pct: float
    pass_threshold_prob: float
    passes: bool
    method: str
    block_size: int


@dataclass(frozen=True)
class RelativeMcResult:
    """Aggregate stats from a relative-MC run (strategy vs benchmark).

    Each path runs BOTH strategy_fn and benchmark_fn on the SAME synthetic
    series. We report the distribution of:

        dd_reduction = strategy_maxdd / benchmark_maxdd

    where maxdd values are negative numbers (max_drawdown returns -0.35 for
    a 35% drawdown). A ratio < 1.0 means the strategy drew down LESS than
    the benchmark on that path. The gate is:

        pass = (median ratio <= median_ratio_gate) AND
               (fraction of paths where strategy MaxDD <= benchmark MaxDD >= p_strategy_better_gate)

    Lesson L17 (added 2026-05-14): absolute P(MaxDD > X%) gates fail for
    long-only equity strategies even when the underlying ITSELF can't pass
    them. Use relative MC for cross-asset momentum, ballast strategies, and
    any other strategy whose thesis is "I add defensive value over the
    underlying" — not "I avoid an absolute drawdown threshold".
    """

    n_paths_completed: int
    strategy_median_maxdd: float  # negative number
    benchmark_median_maxdd: float  # negative number
    strategy_p5_maxdd: float  # worst-case (most negative)
    benchmark_p5_maxdd: float
    median_dd_reduction: float  # strategy_maxdd / benchmark_maxdd  (<1 = strategy better)
    p25_dd_reduction: float  # 25th-percentile reduction
    p75_dd_reduction: float
    p_strategy_better: float  # fraction of paths where strategy MaxDD <= benchmark MaxDD
    median_sharpe_strategy: float
    median_sharpe_benchmark: float
    median_ratio_gate: float
    p_strategy_better_gate: float
    passes: bool
    method: str
    block_size: int
    # Per-path MaxDDs (for dashboard scatter visualization). Kept as tuples
    # of rounded floats so the dataclass stays frozen + hashable.
    strategy_maxdds: tuple[float, ...] = ()
    benchmark_maxdds: tuple[float, ...] = ()


def _resample_indices(
    n_bars: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an array of n_bars resampled indices, drawn as overlapping
    blocks of `block_size`. Each block start is uniformly random in
    [0, n_bars - block_size]. Trailing partial block is truncated.
    """
    if n_bars <= block_size:
        return np.arange(n_bars)
    n_blocks = (n_bars + block_size - 1) // block_size
    starts = rng.integers(0, n_bars - block_size, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return idx[:n_bars]


def _resample_indices_stationary(
    n_bars: int,
    mean_block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis & Romano (1994) STATIONARY bootstrap indices.

    Block lengths are GEOMETRIC (random), not fixed: at each step continue the
    current block with prob (1 - 1/mean_block) or start a fresh uniform-random
    block. Wrap-around (circular) so every bar is equally likely. This preserves
    serial dependence in expectation while keeping the resample stationary --
    the fixed-block scheme's hard boundaries under-represent clustered drawdowns,
    which is what the deploy ruler must capture honestly.
    """
    if n_bars <= 1:
        return np.arange(n_bars)
    p = 1.0 / max(mean_block, 1)
    idx = np.empty(n_bars, dtype=np.int64)
    i = int(rng.integers(0, n_bars))
    for t in range(n_bars):
        idx[t] = i
        if rng.random() < p:
            i = int(rng.integers(0, n_bars))  # start a new block
        else:
            i = (i + 1) % n_bars  # continue the block (circular)
    return idx


def _rebuild_path(
    log_returns: np.ndarray,
    indices: np.ndarray,
    initial_price: float,
) -> np.ndarray:
    """Cumprod of resampled log returns → synthetic price series."""
    resampled = log_returns[indices]
    return initial_price * np.exp(np.cumsum(resampled))


def _run_one_path(
    primary_log_rets: np.ndarray,
    primary_index: pd.DatetimeIndex,
    initial_primary: float,
    extras_log_rets: dict[str, np.ndarray],
    extras_initial: dict[str, float],
    block_size: int,
    bootstrap_method: str,
    strategy_fn: Callable[[pd.DataFrame], pd.Series],
    periods_per_year: int,
    path_seed: int,
    *,
    benchmark_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> tuple[float, float, float | None, float | None] | None:
    """Run one MC path and return per-path metrics.

    Worker function for path-level parallelism. Designed to be cheap to
    pickle (only takes numpy arrays + index + serializable scalars) so it
    works under both joblib's loky backend and a plain ProcessPoolExecutor.

    Returns ``(sharpe, maxdd, bench_sharpe, bench_maxdd)`` where the
    bench_* fields are ``None`` when ``benchmark_fn`` is not supplied
    (used by ``run_relative_block_mc``). Returns ``None`` if the strategy
    raised on the synthetic path or produced too few bars.
    """
    n_bars = len(primary_log_rets)
    rng = np.random.default_rng(path_seed)

    if bootstrap_method == "stationary":
        indices = _resample_indices_stationary(n_bars, block_size, rng)
    else:
        indices = _resample_indices(n_bars, block_size, rng)
    synth_primary = _rebuild_path(primary_log_rets, indices, initial_primary)
    df = pd.DataFrame(
        {"close": synth_primary},
        index=primary_index[: len(synth_primary)],
    )
    if bootstrap_method in ("shared_block", "stationary"):
        # extras follow the SAME resampled indices (cross-series alignment preserved)
        for name, lr in extras_log_rets.items():
            synth = _rebuild_path(lr, indices[: len(lr)], extras_initial[name])
            df[name] = pd.Series(synth, index=df.index[: len(synth)])
    else:  # "block" -- extras independent
        for name, lr in extras_log_rets.items():
            ext_indices = _resample_indices(len(lr), block_size, rng)
            synth = _rebuild_path(lr, ext_indices, extras_initial[name])
            df[name] = pd.Series(synth, index=df.index[: len(synth)])

    try:
        strat_ret = strategy_fn(df)
    except Exception:
        return None
    if len(strat_ret) < 20:
        return None

    sh = float(sharpe(strat_ret, periods_per_year=periods_per_year))
    mdd = float(max_drawdown(strat_ret))

    if benchmark_fn is None:
        return (sh, mdd, None, None)

    try:
        bench_ret = benchmark_fn(df)
    except Exception:
        return None
    if len(bench_ret) < 20:
        return None
    bsh = float(sharpe(bench_ret, periods_per_year=periods_per_year))
    bmdd = float(max_drawdown(bench_ret))
    return (sh, mdd, bsh, bmdd)


def _spawn_path_seeds(seed: int, n_paths: int) -> list[int]:
    """Produce ``n_paths`` independent integer seeds from a master seed.

    Uses ``np.random.SeedSequence.spawn`` so the per-path streams are
    statistically independent and reproducible from the master seed.
    """
    ss = np.random.SeedSequence(seed)
    return [int(child.generate_state(1)[0]) for child in ss.spawn(n_paths)]


def run_block_mc(
    primary_close: pd.Series,
    cfg: McConfig,
    strategy_fn: Callable[[pd.DataFrame], pd.Series],
    *,
    periods_per_year: int,
    seed: int = 42,
    extra_series: dict[str, pd.Series] | None = None,
    n_workers: int = 1,
) -> McResult:
    """Block bootstrap of the primary close (and optional extras at SHARED
    indices for cross-asset strategies).

    Parameters
    ----------
    primary_close:
        The underlying price series the strategy trades.
    cfg:
        McConfig from typology.defaults_for(strategy_class).mc.
    strategy_fn:
        Callable that takes a DataFrame with at least 'close' column
        (and any 'extra_series' keys re-named lowercase) and returns
        a per-bar return Series.
    periods_per_year:
        Annualisation factor for Sharpe.
    seed:
        RNG seed for reproducibility.
    extra_series:
        Optional dict of additional series (e.g. {'bond_close': ...,
        'high': ..., 'low': ...}) that share the primary's bar
        timestamps. For "shared_block" method these are resampled at
        the SAME indices as the primary to preserve correlation.

    Returns:
    -------
    McResult
    """
    primary_close = primary_close.dropna()
    if len(primary_close) < cfg.block_size_bars * 2:
        return McResult(
            n_paths_completed=0,
            median_sharpe=0.0,
            p5_sharpe=0.0,
            p95_sharpe=0.0,
            median_maxdd=0.0,
            p_maxdd_gt_threshold=1.0,
            threshold_pct=cfg.max_dd_threshold_pct,
            pass_threshold_prob=cfg.max_dd_pass_prob,
            passes=False,
            method=cfg.bootstrap_method,
            block_size=cfg.block_size_bars,
        )

    log_returns_primary = np.log(primary_close).diff().dropna().to_numpy()
    initial_primary = float(primary_close.iloc[0])

    extras_log_rets: dict[str, np.ndarray] = {}
    extras_initial: dict[str, float] = {}
    if extra_series:
        # Build the common-valid-index across primary + ALL extras FIRST,
        # then reindex everything to that single shared index. This ensures
        # extras_log_rets[name] has the same length as log_returns_primary
        # (required by the shared_block bootstrap path).
        common_idx = primary_close.index
        for name, s in extra_series.items():
            s_aligned = s.reindex(primary_close.index).dropna()
            common_idx = common_idx.intersection(s_aligned.index)
        if len(common_idx) >= cfg.block_size_bars * 2:
            # Re-align primary first.
            primary_close = primary_close.reindex(common_idx).dropna()
            log_returns_primary = np.log(primary_close).diff().dropna().to_numpy()
            # Now build each extra on the same shared index.
            for name, s in extra_series.items():
                s_aligned = s.reindex(common_idx).dropna()
                if len(s_aligned) < cfg.block_size_bars * 2:
                    continue
                extras_log_rets[name] = np.log(s_aligned).diff().dropna().to_numpy()
                extras_initial[name] = float(s_aligned.iloc[0])

    # Per-path independent seeds (reproducible from master seed).
    path_seeds = _spawn_path_seeds(seed, cfg.n_paths)

    primary_index = primary_close.index

    def _do_path(ps: int):
        return _run_one_path(
            log_returns_primary,
            primary_index,
            initial_primary,
            extras_log_rets,
            extras_initial,
            cfg.block_size_bars,
            cfg.bootstrap_method,
            strategy_fn,
            periods_per_year,
            ps,
        )

    results: list[tuple[float, float, float | None, float | None] | None]
    if n_workers <= 1:
        # Serial path -- matches the pre-parallel behaviour modulo per-path
        # seeding (now deterministic per path rather than draws from a single
        # master RNG; statistically equivalent, more reproducible).
        results = [_do_path(ps) for ps in path_seeds]
    else:
        # Path-level parallelism via joblib (handles closure pickling via
        # cloudpickle under the loky backend).
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
            delayed(_do_path)(ps) for ps in path_seeds
        )

    sharpes = [r[0] for r in results if r is not None]
    maxdds = [r[1] for r in results if r is not None]

    if not sharpes:
        return McResult(
            n_paths_completed=0,
            median_sharpe=0.0,
            p5_sharpe=0.0,
            p95_sharpe=0.0,
            median_maxdd=0.0,
            p_maxdd_gt_threshold=1.0,
            threshold_pct=cfg.max_dd_threshold_pct,
            pass_threshold_prob=cfg.max_dd_pass_prob,
            passes=False,
            method=cfg.bootstrap_method,
            block_size=cfg.block_size_bars,
        )

    sh_arr = np.asarray(sharpes)
    mdd_arr = np.asarray(maxdds)
    p_mdd_gt = float((mdd_arr < -cfg.max_dd_threshold_pct).mean())
    return McResult(
        n_paths_completed=len(sharpes),
        median_sharpe=float(round(np.median(sh_arr), 4)),
        p5_sharpe=float(round(np.quantile(sh_arr, 0.05), 4)),
        p95_sharpe=float(round(np.quantile(sh_arr, 0.95), 4)),
        median_maxdd=float(round(np.median(mdd_arr), 4)),
        p_maxdd_gt_threshold=round(p_mdd_gt, 4),
        threshold_pct=cfg.max_dd_threshold_pct,
        pass_threshold_prob=cfg.max_dd_pass_prob,
        passes=p_mdd_gt <= cfg.max_dd_pass_prob,
        method=cfg.bootstrap_method,
        block_size=cfg.block_size_bars,
    )


def run_relative_block_mc(
    primary_close: pd.Series,
    cfg: McConfig,
    strategy_fn: Callable[[pd.DataFrame], pd.Series],
    benchmark_fn: Callable[[pd.DataFrame], pd.Series],
    *,
    periods_per_year: int,
    seed: int = 42,
    extra_series: dict[str, pd.Series] | None = None,
    median_ratio_gate: float = 0.80,
    p_strategy_better_gate: float = 0.50,
    n_workers: int = 1,
) -> RelativeMcResult:
    """Relative block-bootstrap MC: strategy vs benchmark on the SAME paths.

    Why: an absolute P(MaxDD > X%) gate fails for long-only equity strategies
    because the UNDERLYING ITSELF can't pass it under block-bootstrap (the
    bootstrap shuffles real crisis bars into most synthetic paths). The right
    economic question is: "Does the strategy reduce MaxDD vs the underlying
    on the same realization?" -- which is exactly what a relative gate tests.

    See V3.6 Lesson L17 for the methodology rationale.

    Parameters
    ----------
    primary_close, cfg, strategy_fn, periods_per_year, seed, extra_series:
        As in ``run_block_mc``.
    benchmark_fn:
        Callable mapping the SAME synthetic DataFrame to a per-bar benchmark
        return series. The standard choice is buy-and-hold the primary:

            lambda df: df["close"].pct_change().fillna(0.0)

        Caller can substitute cash (Series of zeros) for market-neutral
        strategies, or any other reference allocation.
    median_ratio_gate:
        PASS gate. Median of ``strategy_maxdd / benchmark_maxdd`` across paths
        must be <= this. 0.8 = strategy reduces MaxDD vs benchmark by >=20%
        on the median path. (Both maxdds are negative; ratio of two negatives
        is positive; smaller ratio = strategy is materially better.)
    p_strategy_better_gate:
        Fraction of paths on which strategy's MaxDD must be no worse than
        the benchmark's. Defaults to 0.5 (strategy is at least as good as
        benchmark on the majority of paths).

    Returns:
    -------
    RelativeMcResult with per-strategy / per-benchmark distribution stats.
    """
    primary_close = primary_close.dropna()
    if len(primary_close) < cfg.block_size_bars * 2:
        return RelativeMcResult(
            n_paths_completed=0,
            strategy_median_maxdd=0.0,
            benchmark_median_maxdd=0.0,
            strategy_p5_maxdd=0.0,
            benchmark_p5_maxdd=0.0,
            median_dd_reduction=1.0,
            p25_dd_reduction=1.0,
            p75_dd_reduction=1.0,
            p_strategy_better=0.0,
            median_sharpe_strategy=0.0,
            median_sharpe_benchmark=0.0,
            median_ratio_gate=median_ratio_gate,
            p_strategy_better_gate=p_strategy_better_gate,
            passes=False,
            method=cfg.bootstrap_method,
            block_size=cfg.block_size_bars,
        )

    log_returns_primary = np.log(primary_close).diff().dropna().to_numpy()
    initial_primary = float(primary_close.iloc[0])

    extras_log_rets: dict[str, np.ndarray] = {}
    extras_initial: dict[str, float] = {}
    if extra_series:
        for name, s in extra_series.items():
            s = s.reindex(primary_close.index).dropna()
            common = primary_close.index.intersection(s.index)
            if len(common) < cfg.block_size_bars * 2:
                continue
            extras_log_rets[name] = np.log(s.reindex(common)).diff().dropna().to_numpy()
            extras_initial[name] = float(s.iloc[0])
        if extras_log_rets:
            common_idx = primary_close.index
            for name in extras_log_rets:
                common_idx = common_idx.intersection(extra_series[name].index)
            primary_close = primary_close.reindex(common_idx).dropna()
            log_returns_primary = np.log(primary_close).diff().dropna().to_numpy()

    primary_index = primary_close.index
    path_seeds = _spawn_path_seeds(seed, cfg.n_paths)

    def _do_path(ps: int):
        return _run_one_path(
            log_returns_primary,
            primary_index,
            initial_primary,
            extras_log_rets,
            extras_initial,
            cfg.block_size_bars,
            cfg.bootstrap_method,
            strategy_fn,
            periods_per_year,
            ps,
            benchmark_fn=benchmark_fn,
        )

    if n_workers <= 1:
        results = [_do_path(ps) for ps in path_seeds]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
            delayed(_do_path)(ps) for ps in path_seeds
        )

    strat_sharpes = [r[0] for r in results if r is not None]
    strat_maxdds = [r[1] for r in results if r is not None]
    bench_sharpes = [r[2] for r in results if r is not None and r[2] is not None]
    bench_maxdds = [r[3] for r in results if r is not None and r[3] is not None]

    if not strat_sharpes:
        return RelativeMcResult(
            n_paths_completed=0,
            strategy_median_maxdd=0.0,
            benchmark_median_maxdd=0.0,
            strategy_p5_maxdd=0.0,
            benchmark_p5_maxdd=0.0,
            median_dd_reduction=1.0,
            p25_dd_reduction=1.0,
            p75_dd_reduction=1.0,
            p_strategy_better=0.0,
            median_sharpe_strategy=0.0,
            median_sharpe_benchmark=0.0,
            median_ratio_gate=median_ratio_gate,
            p_strategy_better_gate=p_strategy_better_gate,
            passes=False,
            method=cfg.bootstrap_method,
            block_size=cfg.block_size_bars,
        )

    strat_mdd_arr = np.asarray(strat_maxdds)
    bench_mdd_arr = np.asarray(bench_maxdds)

    # DD reduction ratio: strategy_maxdd / benchmark_maxdd. Both are
    # negative numbers; ratio is positive. Smaller ratio = strategy did
    # less damage relative to benchmark. Guard against benchmark_maxdd == 0
    # (degenerate path) by clipping.
    safe_bench = np.where(bench_mdd_arr < -1e-6, bench_mdd_arr, -1e-6)
    ratios = strat_mdd_arr / safe_bench

    # Path-wise strategy-no-worse counter. mdd is negative, so "no worse"
    # means strat_mdd >= bench_mdd (less negative).
    strategy_better_mask = strat_mdd_arr >= bench_mdd_arr
    p_better = float(strategy_better_mask.mean())

    median_ratio = float(np.median(ratios))
    passes = (median_ratio <= median_ratio_gate) and (p_better >= p_strategy_better_gate)

    return RelativeMcResult(
        n_paths_completed=len(strat_sharpes),
        strategy_median_maxdd=float(round(np.median(strat_mdd_arr), 4)),
        benchmark_median_maxdd=float(round(np.median(bench_mdd_arr), 4)),
        strategy_p5_maxdd=float(round(np.quantile(strat_mdd_arr, 0.05), 4)),
        benchmark_p5_maxdd=float(round(np.quantile(bench_mdd_arr, 0.05), 4)),
        median_dd_reduction=round(median_ratio, 4),
        p25_dd_reduction=float(round(np.quantile(ratios, 0.25), 4)),
        p75_dd_reduction=float(round(np.quantile(ratios, 0.75), 4)),
        p_strategy_better=round(p_better, 4),
        median_sharpe_strategy=float(round(np.median(strat_sharpes), 4)),
        median_sharpe_benchmark=float(round(np.median(bench_sharpes), 4)),
        median_ratio_gate=median_ratio_gate,
        p_strategy_better_gate=p_strategy_better_gate,
        passes=passes,
        method=cfg.bootstrap_method,
        block_size=cfg.block_size_bars,
        strategy_maxdds=tuple(round(float(x), 4) for x in strat_mdd_arr),
        benchmark_maxdds=tuple(round(float(x), 4) for x in bench_mdd_arr),
    )
