"""trade_ledger.py -- persist closed round-trip trades to the dashboard SQLite.

Why this exists
---------------
The dashboard Trades tab historically read IBKR ``reqExecutions``. But IB only
returns executions to the client that PLACED them (or to whichever client is the
gateway's "Master API Client ID"). The dashboard connects as its own read-only
client (id 30) that never trades, and no master client id is configured, so that
path is **structurally empty** — it can never see the trading client's (id 7)
fills (2026-07-02 finding).

This records each closed position from the trading node's OWN event stream (which
does see every fill) into ``.tmp/titan_dashboard.db::closed_trades``, giving the
Trades tab a persistent, full-history ledger independent of IB's client-scoped
execution API and its ~24h window.

Wiring: a node-level ``events.position.*`` MessageBus subscription in
``titan/utils/global_event_hooks.py`` calls :func:`record_closed_trade` for every
strategy's closes — no per-strategy code. Best-effort + never-raise: a ledger
write must never disrupt trading.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _PROJECT_ROOT / ".tmp" / "titan_dashboard.db"

_NS_PER_SEC = 1_000_000_000

_CREATE_SQL = (
    "CREATE TABLE IF NOT EXISTS closed_trades ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "position_id TEXT NOT NULL, "
    "ts_closed TEXT NOT NULL, "
    "ts_closed_ns INTEGER, "
    "strategy_id TEXT, "
    "symbol TEXT NOT NULL, "
    "direction TEXT, "
    "qty REAL, "
    "entry_price REAL, "
    "exit_price REAL, "
    "realized_pnl REAL, "
    "currency TEXT, "
    "hold_secs REAL, "
    "UNIQUE(position_id, ts_closed_ns))"
)


def _to_float(x: Any) -> float | None:
    """Best-effort float from a NautilusTrader Money/Price/Quantity or number."""
    if x is None:
        return None
    try:
        if hasattr(x, "as_double"):
            return float(x.as_double())
        return float(x)
    except Exception:  # noqa: BLE001
        return None


def _ccy(money: Any) -> str | None:
    try:
        c = getattr(money, "currency", None)
        return str(c) if c is not None else None
    except Exception:  # noqa: BLE001
        return None


def extract_closed_trade(event: Any) -> dict | None:
    """Pull the fields we persist from a NautilusTrader ``PositionClosed`` event.

    Returns a dict, or ``None`` if the event lacks the essentials (instrument /
    symbol). Tolerant of missing attributes (``getattr`` throughout) so it can be
    unit-tested with a lightweight fake event and never trips on an NT version
    that renames a field.
    """
    try:
        instrument_id = getattr(event, "instrument_id", None)
        if instrument_id is None:
            return None
        symbol = str(instrument_id).split(".", 1)[0]
        if not symbol:
            return None
        ts_closed_ns = int(getattr(event, "ts_closed", 0) or 0)
        opened_ns = int(getattr(event, "ts_opened", 0) or 0)
        hold_secs = (
            max(0.0, (ts_closed_ns - opened_ns) / _NS_PER_SEC)
            if ts_closed_ns and opened_ns
            else None
        )
        # `entry` is the entry OrderSide (BUY for a long round-trip).
        direction = "LONG" if "BUY" in str(getattr(event, "entry", "")).upper() else "SHORT"
        pnl = getattr(event, "realized_pnl", None)
        return {
            "position_id": str(getattr(event, "position_id", "") or ""),
            "ts_closed_ns": ts_closed_ns,
            "strategy_id": str(getattr(event, "strategy_id", "") or ""),
            "symbol": symbol,
            "direction": direction,
            "qty": _to_float(getattr(event, "peak_qty", None)),
            "entry_price": _to_float(getattr(event, "avg_px_open", None)),
            "exit_price": _to_float(getattr(event, "avg_px_close", None)),
            "realized_pnl": _to_float(pnl),
            "currency": _ccy(pnl) or (str(getattr(event, "currency", "")) or None),
            "hold_secs": hold_secs,
        }
    except Exception:  # noqa: BLE001
        return None


def write_closed_trade(row: dict, *, db_path: Path | None = None) -> None:
    """Insert one closed-trade row into ``closed_trades`` (idempotent).

    Dedup is on ``UNIQUE(position_id, ts_closed_ns)`` so a redelivered event is a
    no-op. Creates the table on first use. Raises only on a genuine DB error (the
    caller :func:`record_closed_trade` swallows it).
    """
    path = db_path or _DEFAULT_DB
    ts_ns = int(row.get("ts_closed_ns") or 0)
    ts_iso = (
        datetime.fromtimestamp(ts_ns / _NS_PER_SEC, tz=timezone.utc).isoformat() if ts_ns else ""
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as con:
        con.execute(_CREATE_SQL)
        con.execute(
            "INSERT OR IGNORE INTO closed_trades "
            "(position_id, ts_closed, ts_closed_ns, strategy_id, symbol, direction, "
            "qty, entry_price, exit_price, realized_pnl, currency, hold_secs) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row.get("position_id", ""),
                ts_iso,
                ts_ns,
                row.get("strategy_id"),
                row.get("symbol"),
                row.get("direction"),
                row.get("qty"),
                row.get("entry_price"),
                row.get("exit_price"),
                row.get("realized_pnl"),
                row.get("currency"),
                row.get("hold_secs"),
            ),
        )


def record_closed_trade(event: Any, *, db_path: Path | None = None) -> bool:
    """Extract + persist a closed trade from a ``PositionClosed`` event.

    Best-effort: returns True on write, False otherwise, and NEVER raises — a
    ledger write must not disrupt the trading loop.
    """
    try:
        row = extract_closed_trade(event)
        if not row or not row.get("symbol"):
            return False
        write_closed_trade(row, db_path=db_path)
        return True
    except Exception as e:  # noqa: BLE001
        _log.warning("trade_ledger: record failed: %s", e)
        return False
