"""Amortised MC — IS-frozen model state cached once across all MC paths.

Speed-up #2 from the V3.6 speed lever list. The standard ``run_block_mc``
re-invokes ``strategy_fn(synthetic_df)`` on every bootstrap path, which
for model-class strategies (HMM, Kalman, regression) means re-fitting the
model on every path. Under V3.6's IS-frozen rule the model is supposed
to be a constant — re-fitting per path is both wasteful AND semantically
wrong (the deployed strategy fits *once* on real IS, not on each
synthetic IS the bootstrap fabricated).

This module exposes a two-stage protocol:

    prefit(is_data) -> fitted_state       # called ONCE on real IS
    infer(synthetic_df, fitted_state)     # called PER MC path, cheap

For I1 (per-asset HMM gate, 31 instruments × 200 paths × ~5500 bars) the
naïve cost is 200 × 31 = 6,200 HMM fits per cell × 13 cells = 80,600 fits
≈ 50 minutes serial. Amortised: 31 fits ONCE, decode 200 × 31 = 6,200
times per cell — decode is ~50x cheaper than fit, so the cell collapses
to <1 min.

Causality (L04 / L13 / L50):
    - ``prefit`` sees ONLY the original IS data (passed in by the caller).
    - It is NEVER re-invoked on synthetic data; the bootstrap can't leak
      future info into the model fit because the model is frozen before
      bootstrap starts.
    - ``infer`` may use the fitted state to decode each synthetic path
      causally (L50: forward filter, not Viterbi).

This module is a strict superset of the single-callable interface — if
your strategy has no model-fit step you don't need it, ``run_block_mc``
is already optimal.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from titan.research.framework.mc import (
    McResult,
    _rebuild_path,
    _resample_indices,
    _spawn_path_seeds,
)
from titan.research.framework.typology import McConfig
from titan.research.metrics import max_drawdown, sharpe

# Two-stage strategy interface.
PrefitFn = Callable[[pd.DataFrame], Any]
InferFn = Callable[[pd.DataFrame, Any], pd.Series]


def run_block_mc_amortised(
    primary_close: pd.Series,
    cfg: McConfig,
    *,
    prefit: PrefitFn,
    infer: InferFn,
    is_end_idx: int,
    periods_per_year: int,
    seed: int = 42,
    extra_series: dict[str, pd.Series] | None = None,
    n_workers: int = 1,
) -> McResult:
    """Block-bootstrap MC with IS-frozen model state cached once.

    Parameters:
        primary_close:
            Full price series (IS + OOS). Same as ``run_block_mc``.
        cfg:
            ``McConfig`` from the strategy class defaults.
        prefit:
            Called ONCE with a DataFrame containing the IS portion of
            ``primary_close`` (and IS portions of ``extra_series`` keyed
            by their dict keys). Returns an opaque "fitted state" object
            that ``infer`` knows how to consume.
        infer:
            Called PER MC path with ``(synthetic_df, fitted_state)`` →
            per-bar return Series. Must be deterministic given the state
            + path. Must be causal (no peek at future synthetic bars).
        is_end_idx:
            Row index in ``primary_close`` at which IS ends (exclusive).
            ``primary_close.iloc[:is_end_idx]`` is what ``prefit`` sees.
        periods_per_year, seed, extra_series, n_workers:
            As in ``run_block_mc``.

    Returns:
        ``McResult`` — same shape as ``run_block_mc`` so the framework's
        decision matrix consumes it identically.

    Raises:
        ValueError: if ``is_end_idx`` is non-positive or beyond the data.
    """
    primary_close = primary_close.dropna()
    if is_end_idx <= 0 or is_end_idx > len(primary_close):
        raise ValueError(f"is_end_idx={is_end_idx} out of range (1, {len(primary_close)}]")
    if cfg.bootstrap_method == "stationary":
        raise NotImplementedError(
            "stationary bootstrap not implemented (parity with run_block_mc)."
        )
    if len(primary_close) < cfg.block_size_bars * 2:
        return _empty_mc_result(cfg)

    # ── Stage 1: prefit ONCE on the IS portion ────────────────────────────
    is_df = pd.DataFrame({"close": primary_close.iloc[:is_end_idx]})
    if extra_series:
        # Use common-index intersection (L46 fix), restricted to IS.
        common_idx = is_df.index
        aligned_extras: dict[str, pd.Series] = {}
        for name, s in extra_series.items():
            s_aligned = s.reindex(primary_close.index).iloc[:is_end_idx].dropna()
            aligned_extras[name] = s_aligned
            common_idx = common_idx.intersection(s_aligned.index)
        is_df = is_df.loc[common_idx]
        for name, s in aligned_extras.items():
            is_df[name] = s.reindex(common_idx)

    fitted_state = prefit(is_df)

    # ── Stage 2: per-path infer ───────────────────────────────────────────
    # Pre-compute the log-return arrays for bootstrap-path reconstruction
    # (identical to run_block_mc).
    log_returns_primary = np.log(primary_close).diff().dropna().to_numpy()
    initial_primary = float(primary_close.iloc[0])

    extras_log_rets: dict[str, np.ndarray] = {}
    extras_initial: dict[str, float] = {}
    if extra_series:
        common_idx = primary_close.index
        for name, s in extra_series.items():
            s_aligned = s.reindex(primary_close.index).dropna()
            common_idx = common_idx.intersection(s_aligned.index)
        if len(common_idx) >= cfg.block_size_bars * 2:
            primary_close = primary_close.reindex(common_idx).dropna()
            log_returns_primary = np.log(primary_close).diff().dropna().to_numpy()
            for name, s in extra_series.items():
                s_aligned = s.reindex(common_idx).dropna()
                if len(s_aligned) < cfg.block_size_bars * 2:
                    continue
                extras_log_rets[name] = np.log(s_aligned).diff().dropna().to_numpy()
                extras_initial[name] = float(s_aligned.iloc[0])

    primary_index = primary_close.index
    path_seeds = _spawn_path_seeds(seed, cfg.n_paths)

    def _do_path(ps: int) -> tuple[float, float] | None:
        rng = np.random.default_rng(ps)
        n_bars = len(log_returns_primary)
        indices = _resample_indices(n_bars, cfg.block_size_bars, rng)
        synth_primary = _rebuild_path(log_returns_primary, indices, initial_primary)
        df = pd.DataFrame(
            {"close": synth_primary},
            index=primary_index[: len(synth_primary)],
        )
        if cfg.bootstrap_method == "shared_block":
            for name, lr in extras_log_rets.items():
                synth = _rebuild_path(lr, indices[: len(lr)], extras_initial[name])
                df[name] = pd.Series(synth, index=df.index[: len(synth)])
        else:
            for name, lr in extras_log_rets.items():
                ext_indices = _resample_indices(len(lr), cfg.block_size_bars, rng)
                synth = _rebuild_path(lr, ext_indices, extras_initial[name])
                df[name] = pd.Series(synth, index=df.index[: len(synth)])
        try:
            ret = infer(df, fitted_state)
        except Exception:
            return None
        if len(ret) < 20:
            return None
        return (
            float(sharpe(ret, periods_per_year=periods_per_year)),
            float(max_drawdown(ret)),
        )

    results: list[tuple[float, float] | None]
    if n_workers <= 1:
        results = [_do_path(ps) for ps in path_seeds]
    else:
        from joblib import Parallel, delayed

        # NOTE: ``fitted_state`` is captured by closure and shipped to each
        # worker via cloudpickle. If state is *huge* (e.g., per-asset HMMs
        # for thousands of assets) this may dominate IPC — for those use
        # cases run with n_workers=1, since prefit is already done once.
        results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
            delayed(_do_path)(ps) for ps in path_seeds
        )

    sharpes = [r[0] for r in results if r is not None]
    maxdds = [r[1] for r in results if r is not None]

    if not sharpes:
        return _empty_mc_result(cfg)

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


def _empty_mc_result(cfg: McConfig) -> McResult:
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


__all__ = [
    "PrefitFn",
    "InferFn",
    "run_block_mc_amortised",
]
