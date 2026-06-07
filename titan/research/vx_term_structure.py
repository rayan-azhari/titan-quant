"""Cboe VX (VIX futures) term-structure primitives -- free-data path (audit P3-4).

Cboe publishes per-contract daily settlement history for VX futures, free, on
its public CDN -- no account, no API key. (Coverage note: the FREE CDN hosts
contracts from ~Jan 2013 onward; VX inception was 2004 but the 2004-2012 era,
including the GFC, is only on the paid DataShop. See
scripts/download_vx_futures_cboe.py.) This module is the pure (network-free)
logic for turning those per-contract CSVs into a usable term structure for the
VRP crisis-alpha sleeve:

  * expiry calendar -- ``vx_monthly_expiry`` / ``iter_monthly_expiries`` compute
    the standard monthly VX expiry (the Wednesday 30 days before the third
    Friday of the FOLLOWING month), which is also the CDN filename key;
  * parsing -- ``parse_vx_csv`` reads the Cboe schema into a tidy frame;
  * the 2007 rescale -- ``normalize_pre2007_scale`` divides pre-2007-03-26 quotes
    by 10 (before that date VX futures were quoted at 10x the index level), so a
    spliced series is continuous;
  * ``front_month_series`` stitches the nearest-to-expiry contract per day.

The thin HTTP fetch + on-disk orchestration lives in
``scripts/download_vx_futures_cboe.py``; everything here is deterministic and
unit-tested.

Cboe CSV schema (verified live):
    Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest

Gotchas baked in / documented:
  * one file per expiry on the CDN -- there is no single bulk file, hence the
    expiry-calendar enumeration;
  * pre-2007-03-26 quotes are ~10x the modern (index-level) scale;
  * non-trading days carry Settle with zero OHLC; the final (expiry) row may be
    all-zero OHLC with the special settlement in Settle.

Reference: Cboe "US Futures Historical Data" (free) +
https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/
"""

from __future__ import annotations

import calendar
import io
from datetime import date, timedelta

import pandas as pd

# Before this date VX futures were quoted at 10x the index; from it, at 1x.
MULTIPLIER_CHANGE_DATE = date(2007, 3, 26)
VX_INCEPTION = date(2004, 3, 26)  # first VX future traded
_PRICE_COLS = ("open", "high", "low", "close", "settle")


def third_friday(year: int, month: int) -> date:
    """The third Friday of the given month (SPX standard expiration)."""
    fridays = [
        d
        for d in calendar.Calendar().itermonthdates(year, month)
        if d.month == month and d.weekday() == calendar.FRIDAY
    ]
    return fridays[2]


def vx_monthly_expiry(year: int, month: int) -> date:
    """Standard monthly VX final-settlement date for the contract month
    ``(year, month)``: the Wednesday 30 days before the third Friday of the
    FOLLOWING month. This is also the Cboe CDN per-contract filename key.

    (Holiday shifts can move an actual expiry one business day earlier; the
    downloader handles that with a small date-fallback when a file 404s.)
    """
    nxt_year, nxt_month = (year + 1, 1) if month == 12 else (year, month + 1)
    return third_friday(nxt_year, nxt_month) - timedelta(days=30)


def iter_monthly_expiries(start: date, end: date) -> list[date]:
    """All standard monthly VX expiries with settlement date in ``[start, end]``."""
    out: list[date] = []
    year, month = start.year, start.month
    # Walk a couple of months before/after to catch boundary contracts.
    cursor = date(year, month, 1) - timedelta(days=62)
    stop = end + timedelta(days=62)
    while cursor <= stop:
        exp = vx_monthly_expiry(cursor.year, cursor.month)
        if start <= exp <= end:
            out.append(exp)
        cursor = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
    return sorted(set(out))


def parse_vx_csv(text: str) -> pd.DataFrame:
    """Parse one Cboe VX per-contract CSV into a tidy frame indexed by trade
    date, with float price/volume columns and the raw ``futures`` label.
    Returns an empty frame for empty/header-only input.
    """
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return pd.DataFrame()
    rename = {
        "Trade Date": "trade_date",
        "Futures": "futures",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Settle": "settle",
        "Total Volume": "volume",
        "Open Interest": "open_interest",
    }
    df = df.rename(columns=rename)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    keep = ["futures", *_PRICE_COLS, "volume", "open_interest"]
    for c in (*_PRICE_COLS, "volume", "open_interest"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("trade_date").sort_index()
    return df[[c for c in keep if c in df.columns]]


def normalize_pre2007_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Divide pre-2007-03-26 price columns by 10 so a spliced VX series is on a
    single (modern, index-level) scale. Rows on/after the change are untouched.
    Idempotent only if applied once -- call exactly once per raw contract frame.
    """
    if df.empty:
        return df
    out = df.copy()
    mask = out.index < pd.Timestamp(MULTIPLIER_CHANGE_DATE)
    cols = [c for c in _PRICE_COLS if c in out.columns]
    out.loc[mask, cols] = out.loc[mask, cols] / 10.0
    return out


def front_month_series(contracts: dict[date, pd.DataFrame]) -> pd.DataFrame:
    """Stitch a continuous FRONT-MONTH settle series from a ``{expiry: frame}``
    mapping. For each trade date the front month is the nearest contract whose
    expiry is on/after that date. Returns a frame indexed by trade date with
    ``settle``, ``expiry`` and ``days_to_expiry``.
    """
    rows: list[dict] = []
    for expiry, df in contracts.items():
        if df.empty or "settle" not in df.columns:
            continue
        for ts, settle in df["settle"].items():
            td = ts.date() if hasattr(ts, "date") else ts
            if td > expiry:  # ignore stray rows past settlement
                continue
            rows.append({"trade_date": ts, "expiry": expiry, "settle": float(settle)})
    if not rows:
        return pd.DataFrame(columns=["settle", "expiry", "days_to_expiry"])
    stacked = pd.DataFrame(rows)
    # Front month = smallest expiry per trade date.
    idx = stacked.groupby("trade_date")["expiry"].idxmin()
    front = stacked.loc[idx].set_index("trade_date").sort_index()
    front["days_to_expiry"] = [
        (exp - ts.date()).days for ts, exp in zip(front.index, front["expiry"])
    ]
    return front[["settle", "expiry", "days_to_expiry"]]


__all__ = [
    "MULTIPLIER_CHANGE_DATE",
    "VX_INCEPTION",
    "third_friday",
    "vx_monthly_expiry",
    "iter_monthly_expiries",
    "parse_vx_csv",
    "normalize_pre2007_scale",
    "front_month_series",
]
