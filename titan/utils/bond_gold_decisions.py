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

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Shared signal-state file written by BondGoldStrategy after each bar and
# read by the D5 reconciliation check. Always reflects the same z-score the
# live strategy used to make its last trading decision, so D5 never diverges
# from the strategy due to stale parquet data.
_SIGNAL_STATE_FILE = Path(__file__).resolve().parents[2] / ".tmp" / "signal_state.json"
_SIGNAL_STATE_LOCK = threading.Lock()

# Shared hold-state file written by BondGoldStrategy on entry and after each
# daily bar, and read on rehydration. It persists the TRUE elapsed hold
# (``bars_held`` + entry timestamp) per trade instrument so a container
# restart restores the real min-hold progress instead of re-seeding
# ``hold_days``. The old rehydration seeded ``bars_held = hold_days``, which
# made every rehydrated position instantly "past min-hold" — so any restart
# inside a position's min-hold window silently defeated the guard and could
# exit a bar early (observed 2026-06-22: VUSD/EIMI exited on the 4th of 5
# required bars after a restart). Keyed by the trade instrument id string.
_HOLD_STATE_FILE = Path(__file__).resolve().parents[2] / ".tmp" / "bond_gold_hold_state.json"
_HOLD_STATE_LOCK = threading.Lock()

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


def write_signal_state(
    strategy_name: str,
    z: float,
    signal: str,
    *,
    state_file: Path | None = None,
) -> None:
    """Write this strategy's current z-score to the shared signal-state file.

    Called by BondGoldStrategy after each daily bar so D5 always has the
    same z-score the strategy used to make its last trading decision.
    Atomic (tmp-file + replace) and thread-safe. Never raises — failure is
    silently swallowed so a write error never crashes the strategy.
    """
    path = state_file if state_file is not None else _SIGNAL_STATE_FILE
    entry = {"z": round(z, 6), "signal": signal, "ts_ns": time.time_ns()}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _SIGNAL_STATE_LOCK:
            try:
                existing: dict = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            existing[strategy_name] = entry
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            tmp.replace(path)
    except Exception:
        pass


def read_signal_state(
    strategy_name: str,
    *,
    max_age_s: float = 129_600.0,  # 36 h — spans a weekend gap
    state_file: Path | None = None,
) -> float | None:
    """Return the strategy's last written z-score, or None if the entry is
    absent, stale, or the file is unreadable.

    Callers should fall back to parquet-based computation when None is
    returned (e.g. before the first daily bar fires after a container
    restart).
    """
    path = state_file if state_file is not None else _SIGNAL_STATE_FILE
    try:
        data: dict = json.loads(path.read_text(encoding="utf-8"))
        entry = data.get(strategy_name)
        if not entry:
            return None
        age_s = (time.time_ns() - int(entry["ts_ns"])) / 1_000_000_000
        if age_s > max_age_s:
            return None
        return float(entry["z"])
    except Exception:
        return None


def write_hold_state(
    instrument_key: str,
    bars_held: int,
    entry_ts_ns: int,
    *,
    state_file: Path | None = None,
) -> None:
    """Persist the live hold-state (``bars_held`` + entry timestamp) for one
    bond_gold-class trade instrument.

    Called by ``BondGoldStrategy`` on entry and after each daily bar so a
    container restart can restore the TRUE elapsed hold instead of re-seeding
    ``hold_days``. Atomic (tmp-file + replace) and thread-safe. Never raises —
    a write failure is silently swallowed so it can never crash the strategy.
    """
    path = state_file if state_file is not None else _HOLD_STATE_FILE
    entry = {
        "bars_held": int(bars_held),
        "entry_ts_ns": int(entry_ts_ns),
        "ts_ns": time.time_ns(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _HOLD_STATE_LOCK:
            try:
                existing: dict = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            existing[instrument_key] = entry
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            tmp.replace(path)
    except Exception:
        pass


def read_hold_state(
    instrument_key: str,
    *,
    state_file: Path | None = None,
) -> dict | None:
    """Return ``{"bars_held": int, "entry_ts_ns": int}`` for the instrument,
    or None if the entry is absent or the file is unreadable.

    Used by ``BondGoldStrategy`` rehydration to restore the true elapsed hold.
    None means no live state was persisted (e.g. a genuinely pre-existing
    EXTERNAL position this process never opened) — the caller then falls back
    to the conservative ``hold_days`` seed.
    """
    path = state_file if state_file is not None else _HOLD_STATE_FILE
    try:
        data: dict = json.loads(path.read_text(encoding="utf-8"))
        entry = data.get(instrument_key)
        if not entry:
            return None
        return {
            "bars_held": int(entry["bars_held"]),
            "entry_ts_ns": int(entry.get("entry_ts_ns", 0)),
        }
    except Exception:
        return None


def clear_hold_state(
    instrument_key: str,
    *,
    state_file: Path | None = None,
) -> None:
    """Remove the persisted hold-state for one instrument (on exit / close).

    Idempotent and never raises. After clearing, a subsequent rehydration
    finds no state and falls back to the conservative ``hold_days`` seed.
    """
    path = state_file if state_file is not None else _HOLD_STATE_FILE
    try:
        with _HOLD_STATE_LOCK:
            try:
                existing: dict = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            if instrument_key not in existing:
                return
            del existing[instrument_key]
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            tmp.replace(path)
    except Exception:
        pass


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
