"""Loader for Sharadar SEP bulk-export zips (survivorship-free US equity prices).

The Sharadar SEP table (bulk-exported from Nasdaq Data Link) is a ~1 GB zipped
CSV of every US equity's daily prices INCLUDING delisted names -- the
survivorship-free price half of the ic_equity re-audit (the membership half is
``titan.research.framework.universe``). This reads it efficiently by streaming
the inner CSV in chunks and filtering to the requested ticker set before
pivoting to a wide ``date x ticker`` panel, so the ~1200 ever-S&P-500 names fit
comfortably in memory even though the raw table has tens of millions of rows.

SEP schema:
    ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated

``closeadj`` is the split/dividend-adjusted close -- use it for returns.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def _inner_csv_name(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise ValueError("no CSV found inside the SEP zip")
    return names[0]


def load_sep_panel(
    zip_path: str | Path,
    *,
    tickers: Iterable[str] | None = None,
    value_col: str = "closeadj",
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Load a wide ``date x ticker`` price panel from a Sharadar SEP zip.

    ``tickers`` (recommended) restricts the read to a ticker set -- essential
    for the full table. ``value_col`` defaults to ``closeadj`` (adjusted close).
    Returns an empty frame if nothing matches.
    """
    tickset = {t.upper() for t in tickers} if tickers is not None else None
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf, zf.open(_inner_csv_name(zf)) as fh:
        for chunk in pd.read_csv(fh, usecols=["ticker", "date", value_col], chunksize=chunksize):
            if tickset is not None:
                chunk = chunk[chunk["ticker"].str.upper().isin(tickset)]
            if not chunk.empty:
                frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].str.upper()
    panel = df.pivot_table(index="date", columns="ticker", values=value_col, aggfunc="last")
    return panel.sort_index()


def available_tickers(zip_path: str | Path, *, chunksize: int = 2_000_000) -> set[str]:
    """The set of tickers present in the SEP zip (one streaming pass)."""
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf, zf.open(_inner_csv_name(zf)) as fh:
        for chunk in pd.read_csv(fh, usecols=["ticker"], chunksize=chunksize):
            seen.update(chunk["ticker"].str.upper().unique().tolist())
    return seen


__all__ = ["load_sep_panel", "available_tickers"]
