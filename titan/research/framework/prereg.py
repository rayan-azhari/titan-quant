"""Pre-registration provenance + hashing (External Quant Audit 2026-05-29, P4-1).

The DSR / multiple-testing defence rests on N (cells tested), the canonical
cell, and the decision rule being fixed BEFORE the sweep is run. The audit
found 27 of 29 Pre-Reg directives were git-co-committed with their own RETIRE /
DEPLOY result, so there is no timestamp evidence the gate-defining choices were
made blind -- the deflation defence is unverifiable (finding C6).

This module makes pre-registration cryptographically anchored:

  1. Authors put the gate-defining fields in a machine-readable fenced
     ``prereg`` TOML block inside the Pre-Reg directive (strategy_class,
     universe, grid/cells, n_trials, canonical_cell, decision_rule).
  2. :func:`prereg_hash` is a SHA-256 over the CANONICAL JSON of just those
     fields -- prose around the block does not change the hash, and a post-hoc
     edit to N / canonical / thresholds does.
  3. The hash + the directive's git commit + commit time are recorded in the
     result log via :class:`PreRegReceipt`.
  4. A CI check (``scripts/check_prereg_provenance.py``) verifies the result
     commit strictly post-dates the pre-reg commit AND the gate-defining block
     is byte-identical between the two commits.

Grandfathering: directives WITHOUT a ``prereg`` block are legacy prose-only
pre-regs and are NOT subject to the gate. New / re-audit pre-regs include the
block and thereby opt into enforcement -- so this can ship without breaking the
existing corpus (the same way the 2026-05-14 Methodology Audit grandfathered
historical scripts).

Example block (inside a Pre-Reg ``.md``)::

    ```prereg
    schema_version = 1
    strategy_id = "demo_a"
    strategy_class = "CROSS_ASSET_MOMENTUM"
    universe = ["ES", "MES", "ZB"]
    n_trials = 45
    canonical_cell = "J5_canonical"

    [grid]
    lookback = [3, 6, 12]
    vol_target = [0.10, 0.12]

    [decision_rule]
    ci_lo_best = 0.0
    dsr_best = 0.95
    ruin_gate = "V3.8"
    ```
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Fenced-block language tag authors use to mark the machine-readable manifest.
PREREG_FENCE = "prereg"
# Matches a ```prereg ... ``` fenced block (the first one in the document).
_FENCE_RE = re.compile(
    r"^```" + PREREG_FENCE + r"[ \t]*\n(.*?)\n```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

# Required keys in the manifest; absence is a hard error (a half-specified
# pre-reg is worse than a prose-only one because it looks enforced but isn't).
_REQUIRED_KEYS = (
    "schema_version",
    "strategy_id",
    "strategy_class",
    "universe",
    "n_trials",
    "canonical_cell",
    "decision_rule",
)
# The exact set of fields the hash covers (the "gate-defining" content).
_GATE_FIELDS = (
    "schema_version",
    "strategy_id",
    "strategy_class",
    "universe",
    "grid",
    "n_trials",
    "canonical_cell",
    "decision_rule",
)


class PreRegError(ValueError):
    """Raised when a directive's pre-reg manifest is missing or malformed."""


@dataclass(frozen=True)
class PreRegManifest:
    """The gate-defining content of a Pre-Reg directive (parsed from the
    fenced ``prereg`` TOML block).
    """

    schema_version: int
    strategy_id: str
    strategy_class: str
    universe: list[str]
    n_trials: int
    canonical_cell: str
    decision_rule: dict
    grid: dict | None = None

    def gate_fields(self) -> dict:
        """The canonical dict that :func:`prereg_hash` hashes. Only these
        fields define the gate; everything else in the directive is prose.
        """
        return {
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "strategy_class": self.strategy_class,
            "universe": self.universe,
            "grid": self.grid,
            "n_trials": self.n_trials,
            "canonical_cell": self.canonical_cell,
            "decision_rule": self.decision_rule,
        }


@dataclass(frozen=True)
class PreRegReceipt:
    """Provenance receipt to embed verbatim in a result log. Lets a reader
    (or CI) verify the gate-defining choices were fixed before the run.
    """

    prereg_path: str
    prereg_sha256: str
    schema_version: int
    strategy_id: str
    n_trials: int
    canonical_cell: str
    prereg_commit: str | None  # git commit that last touched the manifest, None if dirty/untracked
    prereg_committed_at: str | None  # ISO-8601 commit time, None if unknown
    is_clean: bool  # True iff the directive file has no uncommitted changes

    def to_dict(self) -> dict:
        return {
            "prereg_path": self.prereg_path,
            "prereg_sha256": self.prereg_sha256,
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "n_trials": self.n_trials,
            "canonical_cell": self.canonical_cell,
            "prereg_commit": self.prereg_commit,
            "prereg_committed_at": self.prereg_committed_at,
            "is_clean": self.is_clean,
        }


@dataclass(frozen=True)
class ProvenanceVerdict:
    """Result of the pure post-hoc ordering check (the CI gate's core logic)."""

    ok: bool
    reasons: tuple[str, ...]


# ── Parsing + hashing (pure) ────────────────────────────────────────────────


def extract_prereg_block(text: str) -> str | None:
    """Return the contents of the first ```prereg fenced block, or None if the
    document has no machine-readable manifest (legacy prose-only pre-reg).
    """
    m = _FENCE_RE.search(text)
    return m.group(1) if m else None


def parse_prereg_manifest(source: str | Path) -> PreRegManifest:
    """Parse a Pre-Reg manifest from a directive path or its raw markdown text.

    Raises :class:`PreRegError` if there is no ``prereg`` block or it is
    missing a required key.
    """
    text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else source
    block = extract_prereg_block(text)
    if block is None:
        raise PreRegError(
            "no ```prereg``` manifest block found "
            "(legacy prose-only pre-reg -- not subject to provenance enforcement)"
        )
    try:
        data = tomllib.loads(block)
    except tomllib.TOMLDecodeError as e:  # pragma: no cover - exercised via tests indirectly
        raise PreRegError(f"prereg block is not valid TOML: {e}") from e

    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise PreRegError(f"prereg manifest missing required key(s): {', '.join(missing)}")
    if not isinstance(data["decision_rule"], dict):
        raise PreRegError("prereg 'decision_rule' must be a table/dict")
    if not isinstance(data["universe"], list):
        raise PreRegError("prereg 'universe' must be a list")

    return PreRegManifest(
        schema_version=int(data["schema_version"]),
        strategy_id=str(data["strategy_id"]),
        strategy_class=str(data["strategy_class"]),
        universe=list(data["universe"]),
        n_trials=int(data["n_trials"]),
        canonical_cell=str(data["canonical_cell"]),
        decision_rule=dict(data["decision_rule"]),
        grid=dict(data["grid"]) if isinstance(data.get("grid"), dict) else None,
    )


def prereg_hash(manifest: PreRegManifest) -> str:
    """SHA-256 (hex) over the canonical JSON of the gate-defining fields.

    Canonical = ``sort_keys=True`` + compact separators, so the hash is stable
    across whitespace / key-order differences but changes the instant any
    gate-defining value (N, canonical cell, a threshold, the universe) changes.
    """
    canonical = json.dumps(manifest.gate_fields(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_ordering(
    *,
    same_commit: bool,
    prereg_committed_at: str | None,
    result_committed_at: str | None,
    hash_recorded: str,
    hash_now: str,
) -> ProvenanceVerdict:
    """Pure core of the CI gate: given commit facts, decide if provenance holds.

    Fails if the pre-reg and result share a commit (no blind-selection
    evidence), if the result does not strictly post-date the pre-reg, or if the
    gate-defining block changed since registration (post-hoc edit).
    """
    reasons: list[str] = []
    if same_commit:
        reasons.append("pre-reg and result are in the SAME commit (no blind-selection evidence)")
    if hash_recorded != hash_now:
        reasons.append(
            "gate-defining block changed since registration "
            f"(recorded {hash_recorded[:12]}..., now {hash_now[:12]}...)"
        )
    if prereg_committed_at and result_committed_at:
        if result_committed_at <= prereg_committed_at:
            reasons.append(
                f"result commit ({result_committed_at}) does not post-date "
                f"pre-reg commit ({prereg_committed_at})"
            )
    elif not same_commit:
        # Can't prove ordering without both timestamps; flag rather than pass silently.
        reasons.append("commit timestamps unavailable -- ordering unverifiable")
    return ProvenanceVerdict(ok=not reasons, reasons=tuple(reasons))


# ── Git-backed helpers (thin; isolated for testability) ─────────────────────


def _git(args: list[str], *, repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _repo_root(path: Path) -> Path:
    root = _git(["rev-parse", "--show-toplevel"], repo_root=path.parent)
    return Path(root) if root else path.parent


def git_commit_for_path(path: Path) -> tuple[str | None, str | None]:
    """(commit_sha, ISO-8601 commit time) of the last commit touching ``path``.
    Returns (None, None) if not in a git repo or the file is untracked.
    """
    root = _repo_root(path)
    sha = _git(["log", "-1", "--format=%H", "--", str(path)], repo_root=root)
    when = _git(["log", "-1", "--format=%cI", "--", str(path)], repo_root=root)
    return (sha or None, when or None)


def git_path_is_clean(path: Path) -> bool:
    """True iff ``path`` has no uncommitted (staged or unstaged) changes."""
    root = _repo_root(path)
    status = _git(["status", "--porcelain", "--", str(path)], repo_root=root)
    return status == ""


def build_receipt(path: str | Path) -> PreRegReceipt:
    """Parse the directive's manifest, hash it, and anchor it to git.

    The result-log writer embeds this verbatim. ``prereg_commit`` is ``None``
    and ``is_clean`` is ``False`` when the directive has uncommitted changes --
    a run gated on a dirty pre-reg cannot be anchored and should be rejected by
    the caller (see ``run_audit(strict_prereg=True)``).
    """
    p = Path(path)
    manifest = parse_prereg_manifest(p)
    digest = prereg_hash(manifest)
    clean = git_path_is_clean(p)
    commit, when = git_commit_for_path(p)
    return PreRegReceipt(
        prereg_path=str(p),
        prereg_sha256=digest,
        schema_version=manifest.schema_version,
        strategy_id=manifest.strategy_id,
        n_trials=manifest.n_trials,
        canonical_cell=manifest.canonical_cell,
        prereg_commit=commit if clean else None,
        prereg_committed_at=when if clean else None,
        is_clean=clean,
    )


__all__ = [
    "PREREG_FENCE",
    "PreRegError",
    "PreRegManifest",
    "PreRegReceipt",
    "ProvenanceVerdict",
    "build_receipt",
    "extract_prereg_block",
    "git_commit_for_path",
    "git_path_is_clean",
    "parse_prereg_manifest",
    "prereg_hash",
    "verify_ordering",
]
