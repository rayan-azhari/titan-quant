"""Data-quality manifest + regression gate (audit P1-20, finding H23).

Promotes the data manifest from a descriptive *catalogue* to a blocking
*quality gate*. The catalogue (``scripts/build_data_manifest.py``) records
per-file metadata; this module adds the two things the catalogue lacked:

  1. A content **SHA-256** per file, so a silent overwrite is detectable
     even when the byte size is unchanged. (The motivating incident: IBKR
     silently overwrote the yfinance ``VIX_D.parquet`` -- caught only by a
     manual file-size check. See Methodology Audit 2026-05-14 finding I1.)
  2. A **diff** against the prior manifest that fails on a *regression*:
       * row-count drop          -- data was lost / a download returned fewer bars
       * date-span shrink (start) -- early history disappeared (the GEM 2003->2010 case)
       * date-span shrink (end)   -- recent bars disappeared (a truncated download)
       * source flip              -- a file's provider changed (the VIX_D incident)
       * missing file             -- a previously-present file is gone

Growth (more rows, earlier start, later end, brand-new files) is always fine
-- the gate only blocks *contraction* of known-good data.

Everything here is pure (filesystem reads only, no network / global state) so
the diff logic is exhaustively unit-tested. ``data/`` is gitignored and
regenerable, so the "prior" manifest is the last ``data/manifest.json`` written
on the same machine -- the gate compares a fresh scan against it at download
time (``build_data_manifest.py``) and at live startup (``run_portfolio.py``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MANIFEST_SCHEMA_VERSION = 2
MANIFEST_FILENAME = "manifest.json"
PROVENANCE_FILENAME = "provenance.json"

# ── Regression kinds ───────────────────────────────────────────────────────
ROW_DROP = "row_drop"
SPAN_SHRINK_START = "span_shrink_start"
SPAN_SHRINK_END = "span_shrink_end"
SOURCE_FLIP = "source_flip"
MISSING_FILE = "missing_file"
UNREADABLE = "unreadable"


@dataclass(frozen=True)
class Regression:
    """One blocking data-quality regression of a file vs the prior manifest."""

    kind: str
    file: str
    message: str


def sha256_of_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (1 MiB chunks; never loads it all in RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def read_provenance(data_dir: Path) -> dict[str, str]:
    """Optional ``data/provenance.json`` mapping ``filename -> source``.

    Populated by the download / reconcile scripts (P1-21). Absent -> empty
    dict, so the source-flip check stays dormant until provenance exists;
    row/span/sha checks work regardless. Each value may be a bare source
    string OR a record ``{"source": ..., "roll_rule": ..., ...}`` -- this
    returns just the ``source`` string either way (the manifest only needs
    source for the flip check; the richer record stays in provenance.json).
    """
    path = data_dir / PROVENANCE_FILENAME
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for fname, value in raw.items():
        if isinstance(value, dict):
            src = value.get("source")
            if src:
                out[str(fname)] = str(src)
        elif value:
            out[str(fname)] = str(value)
    return out


def record_provenance(data_dir: Path, updates: dict[str, dict]) -> dict:
    """Merge ``{filename: {"source": ..., "roll_rule": ..., ...}}`` records
    into ``data/provenance.json`` and return the merged mapping (P1-21).

    Download / reconcile scripts call this so the manifest's ``source`` field
    and the P1-20 ``source_flip`` check have data to work against. Merge (not
    replace) so recording one file never drops another's provenance.
    """
    path = data_dir / PROVENANCE_FILENAME
    current: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                current = loaded
        except Exception:
            current = {}
    for fname, record in updates.items():
        current[str(fname)] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    return current


def build_entry(path: Path, *, provenance: dict[str, str] | None = None) -> dict | None:
    """Per-file manifest entry, or None if the file is empty/unreadable.

    Schema v2 = the v1 catalogue fields + ``sha256`` + ``source``.
    """
    stem = path.stem  # e.g. "EUR_USD_H1"
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    bar_count = len(df)
    if bar_count == 0:
        return None
    idx = df.index
    stat = path.stat()
    prov = provenance or {}
    return {
        "symbol": parts[0],
        "timeframe": parts[1],
        "file": path.name,
        "bars": bar_count,
        "first_bar": str(idx[0]),
        "last_bar": str(idx[-1]),
        "columns": list(df.columns),
        "size_kb": round(stat.st_size / 1024, 1),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_of_file(path),
        "source": prov.get(path.name),
    }


def scan_data_dir(data_dir: Path) -> list[dict]:
    """Build entries for every ``*.parquet`` under ``data_dir`` (sorted)."""
    provenance = read_provenance(data_dir)
    entries: list[dict] = []
    for path in sorted(data_dir.glob("*.parquet")):
        entry = build_entry(path, provenance=provenance)
        if entry is not None:
            entries.append(entry)
    return entries


def build_summary(entries: list[dict]) -> dict:
    """Summary stats over the entries (mirrors the v1 catalogue summary)."""
    symbols = sorted({e["symbol"] for e in entries})
    timeframes = sorted({e["timeframe"] for e in entries})
    return {
        "total_files": len(entries),
        "total_symbols": len(symbols),
        "total_bars": sum(e["bars"] for e in entries),
        "total_size_mb": round(sum(e["size_kb"] for e in entries) / 1024, 1),
        "timeframes": {tf: sum(1 for e in entries if e["timeframe"] == tf) for tf in timeframes},
        "symbols": symbols,
    }


def build_manifest(data_dir: Path) -> dict:
    """Full schema-v2 manifest for ``data_dir``."""
    entries = scan_data_dir(data_dir)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": f"{data_dir.name}/",
        "summary": build_summary(entries),
        "files": entries,
    }


def load_manifest(path: Path) -> dict | None:
    """Load a manifest JSON, or None if absent/unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def entries_by_file(manifest: dict | None) -> dict[str, dict]:
    """Index a manifest's ``files`` list by filename."""
    if not manifest:
        return {}
    return {e["file"]: e for e in manifest.get("files", []) if "file" in e}


def _parse_ts(value: object) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(value)
        return None if pd.isna(ts) else ts
    except Exception:
        return None


def diff_entry(prior: dict, current: dict) -> list[Regression]:
    """Regressions of ``current`` vs ``prior`` for one file. Empty == clean.

    Only *contraction* is a regression; growth (more rows, earlier start,
    later end) is fine. A SHA change with unchanged rows/span/source is NOT a
    regression (a legitimate refresh).
    """
    fname = current.get("file") or prior.get("file") or "<unknown>"
    regs: list[Regression] = []

    if current["bars"] < prior["bars"]:
        regs.append(
            Regression(
                ROW_DROP,
                fname,
                f"row count dropped {prior['bars']} -> {current['bars']} "
                f"(-{prior['bars'] - current['bars']} bars)",
            )
        )

    p_start, c_start = _parse_ts(prior.get("first_bar")), _parse_ts(current.get("first_bar"))
    p_end, c_end = _parse_ts(prior.get("last_bar")), _parse_ts(current.get("last_bar"))
    if p_start is not None and c_start is not None and c_start > p_start:
        regs.append(
            Regression(
                SPAN_SHRINK_START,
                fname,
                f"early history lost: start moved {p_start.date()} -> {c_start.date()}",
            )
        )
    if p_end is not None and c_end is not None and c_end < p_end:
        regs.append(
            Regression(
                SPAN_SHRINK_END,
                fname,
                f"recent data lost: end moved {p_end.date()} -> {c_end.date()}",
            )
        )

    p_src, c_src = prior.get("source"), current.get("source")
    if p_src and c_src and p_src != c_src:
        regs.append(Regression(SOURCE_FLIP, fname, f"source changed {p_src} -> {c_src}"))

    return regs


def diff_manifests(
    prior: dict | None,
    current: dict | None,
    *,
    restrict_to: set[str] | None = None,
) -> list[Regression]:
    """All regressions of ``current`` vs ``prior``. No prior -> no regressions.

    ``restrict_to`` (a set of filenames) limits the check to those files; None
    checks every file present in the prior manifest. Files new in ``current``
    are never regressions (additions are fine).
    """
    prior_idx = entries_by_file(prior)
    cur_idx = entries_by_file(current)
    regs: list[Regression] = []
    for fname, p_entry in prior_idx.items():
        if restrict_to is not None and fname not in restrict_to:
            continue
        c_entry = cur_idx.get(fname)
        if c_entry is None:
            regs.append(Regression(MISSING_FILE, fname, "present in prior manifest, now missing"))
            continue
        regs.extend(diff_entry(p_entry, c_entry))
    return regs


def quick_diff_against_disk(
    data_dir: Path,
    prior: dict | None,
    *,
    restrict_to: set[str] | None = None,
) -> list[Regression]:
    """Diff the live ``data_dir`` against a prior manifest, cheaply.

    Hashes each prior file (fast) and only re-reads the parquet (for row/span)
    when the hash changed -- so an unchanged data dir costs a hash sweep, not
    895 parquet reads. Used by the live-startup gate. No prior -> no
    regressions (first run can't regress).
    """
    if not prior:
        return []
    provenance = read_provenance(data_dir)
    regs: list[Regression] = []
    for fname, p_entry in entries_by_file(prior).items():
        if restrict_to is not None and fname not in restrict_to:
            continue
        path = data_dir / fname
        if not path.exists():
            regs.append(Regression(MISSING_FILE, fname, "present in prior manifest, now missing"))
            continue
        prior_sha = p_entry.get("sha256")
        if prior_sha and sha256_of_file(path) == prior_sha:
            continue  # byte-identical -> nothing to check, no parquet read
        c_entry = build_entry(path, provenance=provenance)
        if c_entry is None:
            regs.append(Regression(UNREADABLE, fname, "file is empty or no longer readable"))
            continue
        regs.extend(diff_entry(p_entry, c_entry))
    return regs


def format_regressions(regs: list[Regression]) -> str:
    """Human-readable multi-line summary for logs / CLI output."""
    if not regs:
        return "no data-quality regressions"
    return "\n".join(f"  [{r.kind}] {r.file}: {r.message}" for r in regs)
