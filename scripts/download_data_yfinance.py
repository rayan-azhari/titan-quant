"""download_data_yfinance.py — Pull historical OHLCV from Yahoo Finance.

Downloads daily or hourly bars for ETF instruments using yfinance and writes
Parquet files to data/. Used as a longer-history supplement to Databento
(which only goes back to 2018-05-01 for ARCX.PILLAR).

SPY daily data available from 1993-01-29 (ETF inception).
NOTE: H1 interval (1h) is limited to the last 730 days by Yahoo Finance.

Usage:
    uv run python scripts/download_data_yfinance.py
    uv run python scripts/download_data_yfinance.py --symbols SPY QQQ IWM
    uv run python scripts/download_data_yfinance.py --symbols SPY --start 2000-01-01
    uv run python scripts/download_data_yfinance.py --symbols SPY --interval H1

Output: data/{SYMBOL}_{INTERVAL}.parquet  (DatetimeIndex UTC, columns: open/high/low/close/volume)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

from titan.utils.data_manifest import record_provenance  # noqa: E402

DEFAULT_SYMBOLS = ["SPY"]
DEFAULT_START = "1993-01-01"


def _parse_symbol_arg(arg: str) -> tuple[str, str]:
    """Split a symbol arg into (yahoo_query_ticker, local_save_name).

    Plain ticker:        "SPY"          -> ("SPY", "SPY")
    Mapped form:         "CSPX=CSPX.L"  -> ("CSPX.L", "CSPX")
                         (so query Yahoo for "CSPX.L" but save as "CSPX_D.parquet")

    Lets us pull LSE-listed UCITS ETFs (which Yahoo serves under the .L
    suffix) while keeping the local file name aligned with what the
    strategy code expects.
    """
    if "=" in arg:
        local, yahoo = arg.split("=", 1)
        return yahoo.strip(), local.strip()
    return arg, arg


# Interval mappings: our name -> yfinance name, output suffix
_INTERVAL_MAP = {
    "D": ("1d", "D"),
    "H1": ("1h", "H1"),
    "H4": ("1h", "H1"),  # yfinance has no 4h; user must aggregate manually
    "M5": ("5m", "M5"),
    "M15": ("15m", "M15"),
    "M30": ("30m", "M30"),
}


def download_symbol(symbol: str, start: str, end: str | None, interval: str = "D") -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance for one symbol.

    Args:
        symbol: Ticker (e.g. "SPY").
        start: ISO date string start of range.
        end: ISO date string end of range, or None for today.
        interval: Interval key: "D" (daily), "H1" (hourly), "M5", "M15", "M30".
                  NOTE: yfinance caps intraday history at 730 days (H1) or 60 days (M5/M15/M30).

    Returns:
        DataFrame with UTC DatetimeIndex and open/high/low/close/volume columns.
    """
    import yfinance as yf

    yf_interval, _ = _INTERVAL_MAP.get(interval, ("1d", "D"))
    print(f"  Downloading {symbol} {interval} from Yahoo Finance ({start} to {end or 'today'}) ...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=yf_interval, auto_adjust=True)

    if df.empty:
        print(f"  WARNING: No data returned for {symbol}.")
        return df

    # yfinance returns DatetimeIndex — normalize to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    # Standardise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Keep only OHLCV
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols].copy()
    for col in df.columns:
        df[col] = df[col].astype(float)

    df = df.sort_index().dropna(how="all")
    freq_label = f"{interval} bars"
    print(f"  {symbol}: {len(df)} {freq_label} ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV from Yahoo Finance -> data/*.parquet"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbols to download (default: SPY)",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Start date ISO format (default: 1993-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date ISO format (default: today)",
    )
    parser.add_argument(
        "--interval",
        default="D",
        choices=list(_INTERVAL_MAP.keys()),
        help="Bar interval: D (daily), H1 (hourly, max 730 days), M5/M15/M30 (max 60 days)",
    )
    args = parser.parse_args()

    try:
        import yfinance  # noqa: F401
    except ImportError:
        print("ERROR: yfinance not installed. Run: uv add yfinance")
        sys.exit(1)

    _, suffix = _INTERVAL_MAP.get(args.interval, ("1d", "D"))

    print("=" * 60)
    print("  Yahoo Finance OHLCV Download")
    print(f"  Symbols:  {', '.join(args.symbols)}")
    print(f"  Interval: {args.interval}")
    print(f"  Period:   {args.start} to {args.end or 'today'}")
    print("=" * 60)

    failed: list[str] = []
    prov_updates: dict[str, dict] = {}
    now_iso = datetime.now(timezone.utc).isoformat()
    for sym_arg in args.symbols:
        yahoo_ticker, local_name = _parse_symbol_arg(sym_arg)
        try:
            df = download_symbol(yahoo_ticker, args.start, args.end, args.interval)
            if df.empty:
                failed.append(sym_arg)
                continue
            out_name = f"{local_name}_{suffix}.parquet"
            out_path = DATA_DIR / out_name
            df.to_parquet(out_path)
            print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")
            # P1-21: record provenance so the manifest's source field + the
            # P1-20 source_flip gate have data. Spot ETFs have no roll rule.
            prov_updates[out_name] = {
                "source": "yfinance",
                "query_ticker": yahoo_ticker,
                "interval": args.interval,
                "adjust": "auto_adjust_tr",
                "roll_rule": None,
                "downloaded_utc": now_iso,
            }
        except Exception as exc:
            print(f"  ERROR downloading {sym_arg}: {exc}")
            failed.append(sym_arg)

    if prov_updates:
        record_provenance(DATA_DIR, prov_updates)
        print(f"  Recorded provenance for {len(prov_updates)} file(s) -> data/provenance.json")

    print("=" * 60)
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"  All {len(args.symbols)} symbol(s) downloaded successfully.")

    # Update data manifest
    try:
        import subprocess

        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "build_data_manifest.py")],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass  # Non-critical


if __name__ == "__main__":
    main()
