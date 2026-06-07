"""build_data_manifest.py -- scan data/ and write data/manifest.json.

A machine-readable catalogue of all available market data (symbol, timeframe,
date range, bar count, size, content SHA-256, source) AND a blocking
data-quality gate (audit P1-20): before overwriting the manifest it diffs the
fresh scan against the prior manifest and REFUSES to overwrite if known-good
data regressed -- a row-count drop, a date-span shrink, a source flip, or a
vanished file. That is the guard the silent ``VIX_D`` overwrite (Methodology
Audit I1) needed.

Usage:
    uv run python scripts/build_data_manifest.py            # gate + write
    uv run python scripts/build_data_manifest.py --check    # gate only, no write
    uv run python scripts/build_data_manifest.py --force    # write even on regression

Exit codes: 0 = clean (manifest written, or --check passed); 1 = no data or a
regression was detected without --force. Called automatically by the download
scripts after a refresh, so a download that shrinks a file fails loudly.
Output: data/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from titan.utils.data_manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    build_manifest,
    diff_manifests,
    format_regressions,
    load_manifest,
)

DATA_DIR = PROJECT_ROOT / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build + gate the data manifest.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Diff against the prior manifest and report regressions; do NOT write.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write the manifest even if a regression is detected (operator override).",
    )
    args = parser.parse_args()

    out_path = DATA_DIR / MANIFEST_FILENAME
    print("Scanning data/ for parquet files...")
    current = build_manifest(DATA_DIR)
    entries = current["files"]
    if not entries:
        print("No parquet files found in data/", file=sys.stderr)
        return 1

    prior = load_manifest(out_path)
    regressions = diff_manifests(prior, current)
    if regressions:
        print("\nDATA-QUALITY REGRESSION(S) detected vs prior manifest:", file=sys.stderr)
        print(format_regressions(regressions), file=sys.stderr)

    summary = current["summary"]
    if args.check:
        if regressions and not args.force:
            print("\n--check: FAIL (regressions above).", file=sys.stderr)
            return 1
        print("\n--check: PASS (no data-quality regressions).")
        return 0

    if regressions and not args.force:
        print(
            "\nRefusing to overwrite data/manifest.json -- the new scan regresses "
            "known-good data. Investigate (a bad / partial download?), then re-run "
            "with --force if the change is intentional.",
            file=sys.stderr,
        )
        return 1

    with open(out_path, "w") as f:
        json.dump(current, f, indent=2, default=str)

    print(f"\nManifest written: {out_path}")
    print(f"  Files: {summary['total_files']}")
    print(f"  Symbols: {summary['total_symbols']}")
    print(f"  Total bars: {summary['total_bars']:,}")
    print(f"  Total size: {summary['total_size_mb']} MB")
    print(f"  Timeframes: {summary['timeframes']}")
    if regressions and args.force:
        print("  (written with --force despite regressions above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
