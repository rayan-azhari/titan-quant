#!/usr/bin/env python
"""titan-quant framework demo: watch the validation pipeline make a decision.

Runs the framework end to end on two candidates against a daily price series:

  1. demo_trend  - a simple, causal moving-average rule (no claimed edge).
  2. lucky_noise - a seeded random position series with NO real edge that can
                   look plausible in a single sample.

For each, it computes the deployment evidence the book describes -- a bootstrap
confidence interval on the Sharpe, a Deflated Sharpe over a small parameter
search, an underlying-resampled Monte Carlo tail (P(MaxDD > threshold)), a
held-out sanctuary Sharpe, and a noise-robustness check -- then feeds them to the
framework's total decision function and prints the verdict.

The point is the contrast: the framework is built to REJECT the good-looking-but-
fake candidate on the same gates it uses to bless a real one. Suspicion over
celebration.

    uv run python scripts/validate_demo.py            # uses data/SPY_D.parquet if present
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from titan.research.framework import (
    DecisionInputs,
    StrategyClass,
    decide,
    defaults_for,
    deflated_sharpe,
    run_block_mc,
    slice_sanctuary,
    sr_var_from_sweep,
)
from titan.research.metrics import bootstrap_sharpe_ci, sharpe
from titan.strategies.demo_trend import demo_trend_positions

# Synthetic-data Monte Carlo can produce extreme resampled paths; the resulting
# numpy overflow warnings are cosmetic for this teaching demo.
warnings.filterwarnings("ignore", category=RuntimeWarning)

PPY = 252  # daily bars


def load_close() -> pd.Series:
    """Daily close from data/SPY_D.parquet, or a synthetic GBM fallback."""
    p = Path("data/SPY_D.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        s = df["close"].copy()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        print(f"[data] loaded {len(s)} daily bars from {p}")
        return s.sort_index()
    print("[data] data/SPY_D.parquet not found - using a synthetic GBM series "
          "(run scripts/download_data_yfinance.py for real data).")
    rng = np.random.default_rng(7)
    n = 252 * 16
    idx = pd.bdate_range("2009-01-01", periods=n)
    rets = rng.normal(0.0003, 0.011, n)
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="close")


def _strategy_class() -> StrategyClass:
    for name in ("DAILY_TREND", "CROSS_ASSET_MOMENTUM", "DAILY_MEAN_REVERSION"):
        if hasattr(StrategyClass, name):
            return getattr(StrategyClass, name)
    return next(iter(StrategyClass))


def _noise_robustness(close: pd.Series, positions_fn, *, seed: int = 1) -> tuple[bool, bool]:
    """Re-run under small price-noise injection; report (mean>0, worst>0)."""
    rng = np.random.default_rng(seed)
    sharpes = []
    for _ in range(8):
        noisy = close * (1 + rng.normal(0, 0.001, len(close)))
        pos = positions_fn(noisy)
        sr = sharpe((noisy.pct_change() * pos).dropna(), periods_per_year=PPY)
        sharpes.append(sr)
    arr = np.array(sharpes)
    return bool(arr.mean() > 0), bool(arr.min() > 0)


def evaluate(name: str, close: pd.Series, positions_fn) -> dict:
    sanc = slice_sanctuary(close.to_frame("close"), months=12)
    visible = sanc.visible["close"]
    sanctuary = sanc.sanctuary["close"]

    pos_v = positions_fn(visible)
    strat_v = (visible.pct_change() * pos_v).dropna()
    pos_s = positions_fn(sanctuary)
    strat_s = (sanctuary.pct_change() * pos_s).dropna()

    sr = sharpe(strat_v, periods_per_year=PPY)
    ci_lo, ci_hi = bootstrap_sharpe_ci(strat_v, periods_per_year=PPY, block_size=21)

    # Small parameter "search" to feed the Deflated Sharpe (multiple-testing).
    sweep = []
    for fast, slow in [(10, 50), (20, 100), (30, 150), (40, 200), (15, 75), (25, 125)]:
        try:
            ps = demo_trend_positions(visible, fast=fast, slow=slow)
            sweep.append(sharpe((visible.pct_change() * ps).dropna(), periods_per_year=PPY))
        except ValueError:
            pass
    n_trials = max(len(sweep), 1)
    sr_var = sr_var_from_sweep(sweep) if len(sweep) > 1 else 0.0
    dsr = deflated_sharpe(sr, sr_var_across_trials=sr_var, returns=strat_v, n_trials=n_trials)

    # Underlying-resampled Monte Carlo tail risk.
    mc_cfg = defaults_for(_strategy_class()).mc

    def strategy_fn(df: pd.DataFrame) -> pd.Series:
        col = "close" if "close" in df.columns else df.columns[0]
        return positions_fn(df[col])

    mc = run_block_mc(visible, mc_cfg, strategy_fn, periods_per_year=PPY)

    sanc_sr = sharpe(strat_s, periods_per_year=PPY) if len(strat_s) > 5 else 0.0
    npm, npw = _noise_robustness(visible, positions_fn)

    di = DecisionInputs(
        ci_lo=ci_lo,
        dsr_prob=dsr.dsr_prob,
        p_maxdd_gt_threshold=mc.p_maxdd_gt_threshold,
        pass_threshold_prob=mc.pass_threshold_prob,
        sanctuary_sharpe=sanc_sr,
        noise_passes_mean=npm,
        noise_passes_worst=npw,
    )
    verdict = decide(di).verdict
    return {
        "name": name, "oos_sharpe": sr, "ci_lo": ci_lo, "dsr_prob": dsr.dsr_prob,
        "p_maxdd": mc.p_maxdd_gt_threshold, "sanctuary_sharpe": sanc_sr,
        "verdict": getattr(verdict, "name", str(verdict)),
    }


def main() -> int:
    close = load_close()

    # Market-neutral random long/short: zero expected edge, and (unlike a random
    # long-only series) it cannot free-ride the market's drift, so the framework
    # should reject it on the CI_lo gate.
    rng = np.random.default_rng(42)
    noise_pos = pd.Series(rng.choice([-1.0, 1.0], size=len(close)), index=close.index).shift(1).fillna(0.0)

    candidates = {
        "demo_trend": lambda s: demo_trend_positions(s, 20, 100),
        "lucky_noise": lambda s: noise_pos.reindex(s.index).fillna(0.0),
    }

    rows = [evaluate(n, close, fn) for n, fn in candidates.items()]

    print("\n" + "=" * 92)
    print(f"{'candidate':<14}{'OOS Sharpe':>12}{'CI_lo':>10}{'DSR P(>0)':>12}"
          f"{'P(MaxDD>thr)':>14}{'sanct Sharpe':>14}{'  verdict':>14}")
    print("-" * 92)
    for r in rows:
        print(f"{r['name']:<14}{r['oos_sharpe']:>12.2f}{r['ci_lo']:>10.2f}{r['dsr_prob']:>12.2f}"
              f"{r['p_maxdd']:>14.2%}{r['sanctuary_sharpe']:>14.2f}{r['verdict']:>14}")
    print("=" * 92)
    print("\nThe gate that matters: a 95% Sharpe lower bound (CI_lo) <= 0 means the edge is not "
          "statistically distinguishable from zero -> the candidate is unconfirmed, regardless of how\n"
          "good its point estimate looks. That is how a fake survives a single sample but not the framework.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
