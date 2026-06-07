#!/usr/bin/env python
"""Redaction gate for the PUBLIC companion repo.

Scans a directory tree for anything that must never appear in the public release:
real account IDs, broker secrets/tokens, the live instrument shortlist, and the
private strategy/bundle code-names. Exits non-zero if any leak is found.

Run at publish time (over the export) AND in the public repo's CI so a future PR
cannot reintroduce a leak.

    python scripts/check_public_redaction.py [ROOT]   # default ROOT = "."
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# GENERIC / structural leak patterns only. These reveal nothing project-specific,
# so this file is safe to ship in the public repo (it runs in public CI to stop a
# future contributor committing a secret or account id). Project-specific name
# patterns (live strategy / instrument code-names) are applied at PUBLISH time by
# the private publisher (scripts/publish_clean.py), which is never shipped.
LEAK_PATTERNS = [
    (re.compile(r"\bDU[A-Z]?\d{6,}\b"), "broker account id (DU…)"),
    (re.compile(r"\b\d{8,9}\b(?=.*conid|.*conId)", re.I), "possible IB conId"),
    (re.compile(r"xoxb-[A-Za-z0-9-]{8,}"), "slack bot token"),
    (re.compile(r"https://hooks\.slack\.com/services/\S+"), "slack webhook"),
    (re.compile(r"\bbot\d{6,}:[A-Za-z0-9_-]{20,}"), "telegram bot token"),
]

# KEY=VALUE credential check, but allow obvious placeholders.
CRED_KEY = re.compile(
    r"(?i)\b(tws_userid|tws_password|password|passwd|api[_-]?key|secret[_-]?key|access[_-]?token)\b\s*[=:]\s*[\"']?([^\s\"'#]+)"
)
# A value is a placeholder if it STARTS with one of these markers (prefix match),
# or is empty/very short/numeric/boolean.
PLACEHOLDER = re.compile(
    r"(?i)^(your[_-]|xxx|<|\$\{|changeme|example|placeholder|paper[_-]|dummy|redacted|"
    r"none|true|false|\d+$|.{0,2}$)"
)

SKIP_DIRS = {".git", "__pycache__", "site", ".venv", "node_modules", ".mypy_cache", ".ruff_cache"}
SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".parquet", ".lock", ".ico", ".woff", ".woff2"}
SKIP_NAMES = {"check_public_redaction.py"}  # this file holds the patterns itself


def scan(root: Path) -> list[tuple[str, int, str, str]]:
    hits: list[tuple[str, int, str, str]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts) or p.suffix.lower() in SKIP_SUFFIX or p.name in SKIP_NAMES:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(p.relative_to(root))
        for i, line in enumerate(text.splitlines(), 1):
            for pat, label in LEAK_PATTERNS:
                if pat.search(line):
                    hits.append((rel, i, label, line.strip()[:120]))
            if not p.name.endswith(".example"):  # .env.example holds placeholder keys by design
                m = CRED_KEY.search(line)
                if m and not PLACEHOLDER.match(m.group(2)):
                    hits.append((rel, i, f"credential value ({m.group(1)})", line.strip()[:120]))
    return hits


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    hits = scan(root)
    if not hits:
        print(f"[redaction] PASS — no leaks found under {root}")
        return 0
    print(f"[redaction] FAIL — {len(hits)} potential leak(s) under {root}:\n", file=sys.stderr)
    for rel, ln, label, snippet in hits:
        print(f"  {rel}:{ln}  [{label}]  {snippet}", file=sys.stderr)
    print("\nRefusing to publish. Sanitise the above (or extend the allowlist deliberately).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
