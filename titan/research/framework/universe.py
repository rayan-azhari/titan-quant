"""Survivorship-free point-in-time universe resolver (audit data-gate / C5).

The ic_equity sleeve was audited on "best-3-of-482 CURRENT S&P constituents" --
a survivorship bias: today's index members exclude every name that was dropped
or delisted, so a backtest over them silently conditions on survival and
inflates the apparent edge. A correct re-audit needs the universe AS IT WAS on
each historical rebalance date, INCLUDING names later removed.

This is the provider-agnostic ingestion primitive. Feed it a membership table
(``ticker, date_added, date_removed``) -- from Norgate / CRSP / any historical
-constituents source -- and it yields the exact universe in effect on any date,
or the per-rebalance-date universe for a backtest. A ticker may leave and
rejoin (multiple intervals); ``date_removed = None`` means still a member.

Membership convention: half-open ``[date_added, date_removed)`` -- a name added
on D is a member from D onward, and on its removal date R it is NO LONGER a
member (last membership day is R-1). This matches how index providers timestamp
"effective" changes.

It does NOT download prices (that's the provider's job + ``scripts/download_*``);
it makes whatever membership history you acquire usable without survivorship
bias. Pair the resulting per-date universe with the price parquets when building
the cross-sectional ic_equity re-audit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Membership:
    """One contiguous membership interval for a ticker.

    ``date_removed = None`` means the ticker is still a member. A ticker that
    left and later rejoined the index is represented by multiple ``Membership``
    rows with the same ``ticker``.
    """

    ticker: str
    date_added: date
    date_removed: date | None = None

    def covers(self, as_of: date) -> bool:
        """True if this interval includes ``as_of`` under half-open semantics."""
        if as_of < self.date_added:
            return False
        return self.date_removed is None or as_of < self.date_removed


def universe_as_of(memberships: Iterable[Membership], as_of: date) -> list[str]:
    """The sorted, de-duplicated set of tickers that were members on ``as_of``.

    A delisted name is correctly included on dates within its historical
    membership interval even though it is absent from the current universe --
    that is the whole point (no survivorship bias).
    """
    return sorted({m.ticker for m in memberships if m.covers(as_of)})


def point_in_time_universe(
    memberships: Iterable[Membership], dates: Iterable[date]
) -> dict[date, list[str]]:
    """Map each rebalance date to the universe in effect on that date."""
    members = list(memberships)  # allow a generator to be reused per date
    return {d: universe_as_of(members, d) for d in dates}


def validate_memberships(memberships: Iterable[Membership]) -> list[str]:
    """Return human-readable warnings for suspicious membership data:

    * an interval whose ``date_removed`` is on/before its ``date_added``;
    * overlapping intervals for the same ticker (a data error -- a ticker can
      rejoin, but its intervals should not overlap).

    Returns an empty list when the table is clean. Does not raise -- the caller
    decides whether to treat warnings as fatal.
    """
    warnings: list[str] = []
    by_ticker: dict[str, list[Membership]] = {}
    for m in memberships:
        if m.date_removed is not None and m.date_removed <= m.date_added:
            warnings.append(
                f"{m.ticker}: date_removed {m.date_removed} <= date_added {m.date_added}"
            )
        by_ticker.setdefault(m.ticker, []).append(m)
    for ticker, rows in by_ticker.items():
        ordered = sorted(rows, key=lambda r: r.date_added)
        for prev, nxt in zip(ordered, ordered[1:]):
            prev_end = prev.date_removed
            if prev_end is None or nxt.date_added < prev_end:
                warnings.append(
                    f"{ticker}: overlapping intervals "
                    f"(added {nxt.date_added} before prior end {prev_end})"
                )
    return warnings


def _coerce_date(value: object) -> date | None:
    """Parse a cell into a ``date``; empty / NaN / NaT / None -> None (still active)."""
    # pd.isna cleanly catches None, float NaN, and pandas NaT (which is itself a
    # datetime subclass, so it must be filtered before the isinstance checks).
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s or s.lower() in {"nan", "nat", "none", ""}:
        return None
    return pd.Timestamp(s).date()


def load_memberships(
    path: str | Path,
    *,
    ticker_col: str = "ticker",
    added_col: str = "date_added",
    removed_col: str = "date_removed",
) -> list[Membership]:
    """Load a provider-agnostic membership table from CSV or Parquet.

    Required columns (override the names if your provider differs):
    ``ticker``, ``date_added``, ``date_removed`` (the last may be blank/NaN for
    still-active names). One row per contiguous interval.
    """
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)
    missing = {ticker_col, added_col} - set(df.columns)
    if missing:
        raise ValueError(f"membership table missing required column(s): {sorted(missing)}")
    out: list[Membership] = []
    for _, row in df.iterrows():
        added = _coerce_date(row[added_col])
        if added is None:
            raise ValueError(f"row for {row[ticker_col]!r} has no date_added")
        removed = _coerce_date(row[removed_col]) if removed_col in df.columns else None
        out.append(Membership(str(row[ticker_col]).strip(), added, removed))
    return out


__all__ = [
    "Membership",
    "universe_as_of",
    "point_in_time_universe",
    "validate_memberships",
    "load_memberships",
]
