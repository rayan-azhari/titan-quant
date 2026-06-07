"""audit_codebase_methodology.py -- programmatic gap detector.

Pre-registered in directives/Methodology Audit & Unified Framework 2026-05-14.md
(catalogue §1). Scans the codebase for the documented anti-patterns and
reports findings by severity.

Detection patterns (severity codes from the directive):

    A1  hard-coded sqrt(252) outside shared metrics                    (minor)
    A2  filter-then-annualise: rets[rets != 0] before sharpe           (minor)
    B1  position * bar_returns without .shift(1) on positions          (major)
    B4  .expanding() over close                                        (minor)
    B5  .ffill of higher-TF onto lower-TF (look for ffill after shift) (major)
    A3a periods_per_year omitted on Sharpe call                        (major)
    G1  ad-hoc Sharpe formula without periods_per_year argument        (major)
    C4  global z-score instead of is_frozen_zscore                     (minor)

Output: a parquet at .tmp/reports/methodology_audit/findings_{stamp}.parquet
plus a console summary grouped by severity.

This script is intentionally line-oriented (no AST parsing). It's a
sieve, not a proof -- false positives are expected and the operator
reviews them. It exists to give a baseline catalogue of risk in the
codebase that a strategy audit must inspect.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files / dirs to exclude (auto-generated, tests, third-party, the audit
# script itself, the framework's own implementation).
EXCLUDE_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".tmp",
    "node_modules",
    "data",
    "models",
    "resources",
}
EXCLUDE_FILES = {
    "audit_codebase_methodology.py",
    # Framework modules deliberately use the patterns they're standardising
    # (e.g. sharpe call sites with periods_per_year). Excluding the
    # framework itself keeps the audit signal-to-noise high.
    "framework/typology.py",
    "framework/wfo.py",
    "framework/sanctuary.py",
    "framework/mc.py",
    "framework/dsr.py",
    "framework/decision.py",
    "framework/__init__.py",
    # Tests + their helpers
    "test_framework_synthetic.py",
    "test_ic_census_lib.py",
    # The shared metrics module DEFINES the corrected Sharpe -- it's
    # allowed to mention sqrt(periods_per_year) etc.
    "metrics.py",
}


@dataclass
class Finding:
    severity: str  # "major" | "minor"
    pattern_id: str
    file: str
    line: int
    excerpt: str
    note: str = ""


# Pattern definitions -- each is a list of regex's; if any matches, the
# pattern_id is reported with the supplied severity + note.

PATTERNS: list[tuple[str, str, list[str], str]] = [
    # (pattern_id, severity, [regexes], note)
    (
        "A1_hardcoded_sqrt_252",
        "minor",
        [r"\bsqrt\s*\(\s*252\s*\)", r"np\.sqrt\s*\(\s*252\s*\)", r"math\.sqrt\s*\(\s*252\s*\)"],
        "Hard-coded sqrt(252) outside shared metrics. Use titan.research.metrics.sharpe with explicit periods_per_year.",
    ),
    (
        "A2_filter_then_annualise",
        "minor",
        [
            r"\[[a-zA-Z_]+\s*!=\s*0(?:\.0)?\s*\]\s*\.",
            r"rets\s*\[\s*rets\s*!=\s*0",
            r"returns\s*\[\s*returns\s*!=\s*0",
        ],
        "Filter-then-annualise (rets[rets!=0]) overstates Sharpe by sqrt(1/active_ratio).",
    ),
    (
        "B1_pos_times_ret_no_shift",
        "major",
        [
            r"(?<!\.shift\(1\)\s)\bpositions?\s*\*\s*(?:bar_)?returns?\b",
            r"(?<!\.shift\(1\)\.fillna\(0\.0\)\s)\bpos\s*\*\s*(?:bar_)?ret(?:s|urns)?\b",
        ],
        "position * bar_return without .shift(1). Same-bar look-ahead if position uses close[t] and return is t-1 -> t.",
    ),
    (
        "B4_expanding_close",
        "minor",
        [
            r"\.expanding\([^)]*\)\.std\(\)",
            r"\.expanding\([^)]*\)\.mean\(\)",
            r"close\.expanding\(",
        ],
        "expanding() over close introduces look-ahead variance. Use is_frozen_zscore or rolling_zscore.",
    ),
    (
        "B5_ffill_after_reindex",
        "major",
        [r"\.reindex\([^)]*\)\.ffill\(", r"\.ffill\(\)\.reindex\("],
        "ffill of higher-TF onto lower-TF without prior .shift(1) is the EUR/USD MTF +1.94 Sharpe bug pattern.",
    ),
    (
        "A3_periods_per_year_default",
        "major",
        [r"sharpe\s*\([^)]*\)\s*(?!\*)"],
        "Sharpe call without periods_per_year kwarg. Every call must pass explicit periods_per_year.",
    ),
    (
        "G1_local_sharpe_no_kwarg",
        "minor",
        [
            r"def\s+(_?\w*sharpe)\s*\(\s*[^,)]+\s*\):",
            r"def\s+(_?\w*sharpe)\s*\(\s*[^,)]+\s*\)\s*->",
        ],
        "Local Sharpe definition without periods_per_year argument. Reimplements rather than imports the shared metric.",
    ),
    (
        "C4_global_zscore",
        "minor",
        [
            r"\([^)]*\.mean\(\)\s*\)\s*/\s*[^)]*\.std\(\)",
            r"\(\s*\w+\s*-\s*\w+\.mean\(\)\)\s*/\s*\w+\.std\(\)",
        ],
        "Global (full-series) z-score is look-ahead. Use rolling_zscore or is_frozen_zscore.",
    ),
]


def _should_scan(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return False
    rel = path.relative_to(PROJECT_ROOT)
    rel_str = str(rel).replace("\\", "/")
    for ex in EXCLUDE_FILES:
        if rel_str.endswith(ex):
            return False
    return True


def scan_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return findings
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for pattern_id, severity, regexes, note in PATTERNS:
            # Some patterns need a NEGATIVE check that's hard to encode
            # in a single regex (e.g. A3: "sharpe(" without "periods_per_year").
            if pattern_id == "A3_periods_per_year_default":
                if re.search(r"\bsharpe\s*\(", line) and "periods_per_year" not in line:
                    # Avoid false positives from function definitions / comments
                    if re.search(r"def\s+\w*sharpe", line) or "from " in line or "import" in line:
                        continue
                    findings.append(
                        Finding(
                            severity=severity,
                            pattern_id=pattern_id,
                            file=str(path.relative_to(PROJECT_ROOT)),
                            line=i,
                            excerpt=stripped[:200],
                            note=note,
                        )
                    )
                continue
            for rx in regexes:
                if re.search(rx, line):
                    findings.append(
                        Finding(
                            severity=severity,
                            pattern_id=pattern_id,
                            file=str(path.relative_to(PROJECT_ROOT)),
                            line=i,
                            excerpt=stripped[:200],
                            note=note,
                        )
                    )
                    break
    return findings


def scan_all(paths: list[str] | None = None) -> list[Finding]:
    """Programmatic entry point -- runs the scanner and returns findings.

    Used by ``tests/test_methodology_audit_baseline.py`` to enforce that
    finding counts only DECREASE over time. The CLI ``main`` is a thin
    wrapper around this.
    """
    roots = paths if paths is not None else ["research", "titan", "scripts"]
    all_findings: list[Finding] = []
    for root in roots:
        root_path = PROJECT_ROOT / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.py"):
            if not _should_scan(path):
                continue
            all_findings.extend(scan_file(path))
    return all_findings


def counts_by_pattern(findings: list[Finding]) -> dict[str, int]:
    """Aggregate findings into ``{pattern_id: count}``."""
    out: dict[str, int] = {}
    for f in findings:
        out[f.pattern_id] = out.get(f.pattern_id, 0) + 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Codebase methodology audit")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=["research", "titan", "scripts"],
        help="Roots to scan (relative to project root).",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Write findings to .tmp/reports/methodology_audit/findings_{stamp}.parquet",
    )
    parser.add_argument(
        "--severity",
        default=None,
        choices=["major", "minor"],
        help="Filter findings to one severity.",
    )
    args = parser.parse_args()

    all_findings: list[Finding] = []
    for root in args.paths:
        root_path = PROJECT_ROOT / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.py"):
            if not _should_scan(path):
                continue
            all_findings.extend(scan_file(path))

    if args.severity:
        all_findings = [f for f in all_findings if f.severity == args.severity]

    # Console summary
    print("=" * 90)
    print("  CODEBASE METHODOLOGY AUDIT")
    print(f"  Scanned: {', '.join(args.paths)}")
    print(f"  Found {len(all_findings)} potential issues")
    print("=" * 90)
    print()
    by_pattern: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for f in all_findings:
        by_pattern[f.pattern_id] = by_pattern.get(f.pattern_id, 0) + 1
        by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
    print("  Findings by severity:")
    for sev in ("major", "minor"):
        n = by_severity.get(sev, 0)
        print(f"    {sev:<8} {n:>4}")
    print()
    print("  Findings by pattern:")
    for pid in sorted(by_pattern.keys()):
        print(f"    {pid:<35} {by_pattern[pid]:>4}")
    print()

    # Per-finding detail (capped at 40 for readability)
    print("  Top findings (first 40):")
    for f in all_findings[:40]:
        print(f"    [{f.severity}] {f.pattern_id} -- {f.file}:{f.line}")
        print(f"          {f.excerpt[:140]}")

    if len(all_findings) > 40:
        print(f"    ... and {len(all_findings) - 40} more.")

    if args.write_parquet and all_findings:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / ".tmp" / "reports" / "methodology_audit"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"findings_{stamp}.parquet"
        df = pd.DataFrame([f.__dict__ for f in all_findings])
        df.to_parquet(out_path, index=False)
        print(f"\n  Wrote {out_path.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
