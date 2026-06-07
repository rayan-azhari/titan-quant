"""Pure-Python decision primitives for the bond_gold strategy family.

Mirrors the entry/exit logic in
``titan/strategies/bond_gold/strategy.py::BondGoldStrategy._run_signal``
in a side-effect-free form so it can be reused by:

  - ``scripts/replay_audit.py``  (Tier 2.5: weekly backtest-vs-live diff)
  - ``titan/strategies/reconciliation/strategy.py`` (Tier 2.1: real-time
    shadow-decision branch in the watchdog)

Kept in lockstep with production by structural review — when the live
``BondGoldStrategy`` decision logic changes, this module must change
too. Both ``tests/test_replay_audit.py`` and the reconciliation D5
tests exercise this module directly.

See ``directives/Operational Robustness Framework 2026-05-12.md`` for
the broader operational-robustness framework these primitives serve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ── Per-strategy config (mirrors STRATEGY_REGISTRY entries) ──────────


@dataclass
class BondGoldDecisionConfig:
    """Configuration for one bond_gold-class strategy decision evaluator.

    Each entry must match the corresponding ``BondGoldConfig`` in
    ``scripts/run_portfolio.py::STRATEGY_REGISTRY`` exactly. If the live
    config changes, this dataclass / the per-strategy values that use
    it must change too.
    """

    name: str
    trade_symbol: str  # e.g. "VUSD" — what gets traded and audited against fills
    signal_ticker: str  # e.g. "IHYG" — parquet basename for the signal close series
    lookback: int  # signal momentum lookback in days
    threshold: float
    hold_days: int
    zscore_window: int = 504


# Live-deployed configs. Update in lockstep with run_portfolio.py
# STRATEGY_REGISTRY when the live config changes.
LIVE_CONFIGS: dict[str, BondGoldDecisionConfig] = {
    "demo_xa3": BondGoldDecisionConfig(
        name="demo_xa3",
        trade_symbol="CSPX",
        signal_ticker="IHYU",
        lookback=10,
        threshold=0.50,
        hold_days=10,
    ),
    "demo_xa2": BondGoldDecisionConfig(
        name="demo_xa2",
        trade_symbol="VUSD",
        signal_ticker="IHYG",
        lookback=5,
        threshold=0.25,
        hold_days=5,
    ),
    "demo_xa1": BondGoldDecisionConfig(
        name="demo_xa1",
        trade_symbol="EIMI",
        signal_ticker="IHYG",
        lookback=5,
        threshold=0.25,
        hold_days=5,
    ),
}


# ── Pure functions ───────────────────────────────────────────────────


def compute_z_score(
    closes: list[float],
    *,
    lookback: int,
    zscore_window: int,
) -> float | None:
    """Replicates the z-score computation in BondGoldStrategy._run_signal.

    Returns the z-score of the last bar's lookback-period log-momentum,
    normalised on a trailing zscore_window of historical momentums.
    Returns None if insufficient bars or degenerate (zero-variance)
    series.
    """
    if len(closes) < lookback + 10:
        return None
    mom = math.log(closes[-1] / closes[-1 - lookback])
    first_valid = lookback + 10
    all_moms = [math.log(closes[i] / closes[i - lookback]) for i in range(first_valid, len(closes))]
    if len(all_moms) < 20:
        return None
    window = min(zscore_window, len(all_moms))
    window_moms = np.asarray(all_moms[-window:], dtype=float)
    mu = float(window_moms.mean())
    sigma = float(window_moms.std())
    if sigma < 1e-8:
        return None
    return (mom - mu) / sigma


def expected_action(
    z: float,
    *,
    is_long: bool,
    bars_held: int,
    threshold: float,
    hold_days: int,
) -> str:
    """Mirrors the entry / exit branch in BondGoldStrategy._run_signal.

    Returns ``"entry"``, ``"exit"``, or ``"hold"``.
    Post-fix logic: ``is_long`` derived from ``signed_qty > 0``, not
    ``str(position.side)``.
    """
    # Exit: long, past min-hold, z drops below threshold
    if is_long and bars_held >= hold_days and z <= threshold:
        return "exit"
    # Entry: not long, z above threshold
    if not is_long and z > threshold:
        return "entry"
    return "hold"


def load_signal_closes_from_parquet(ticker: str, data_dir: Path) -> list[float] | None:
    """Load the signal-instrument parquet and return its sorted close
    series as a Python list (matches the in-memory format the live
    strategy buffers internally).

    Returns None if the file is missing — caller should treat as "shadow
    cannot evaluate this strategy" and skip rather than fail.
    """
    path = data_dir / f"{ticker}_D.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path).sort_index()
    except Exception:
        return None
    if "close" not in df.columns:
        return None
    return df["close"].astype(float).tolist()
