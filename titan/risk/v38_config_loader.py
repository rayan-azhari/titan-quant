"""Parse V3.8 envelope config from strategy TOML files.

Per `directives/PRM Integration V3.8 2026-05-24.md` §3.1 + §3.4. Strategies
that opt into the V3.8 envelope add an optional `[v38_envelope]` table to
their TOML; strategies that omit the table get the default (disabled,
V3.7-compatibility).

Schema
======

    [v38_envelope]
    enabled = false              # default; set true after V3.8 re-audit PASSES
    mode = "shadow"              # "shadow" (log only) | "live" (enforce)

Both fields are optional. `enabled = false` is the V3.7-compatibility
default. When `enabled = true`, `mode` controls whether the envelope
enforces rejections (`live`) or only logs `would_have_rejected` events
(`shadow`).

Runtime usage
=============

    >>> from titan.risk.v38_config_loader import load_v38_config_for_strategy
    >>> cfg = load_v38_config_for_strategy("config/demo_b.toml")
    >>> prm.set_strategy_v38_config("gem", cfg)

The live runtime (each `scripts/run_live_*.py`) calls this at startup
after `register_strategy` so the PRM has the per-strategy envelope
config before any bars arrive.

Hot reload
==========

Per directive §3.4 the operator updates a config file + sends SIGHUP
to the container. The `scripts/reload_v38_config.py` CLI reads all
configs and prints the diff, then operator restarts the container (or
the SIGHUP handler -- see future work -- re-loads).

No defaults in this module duplicate defaults in `v38_envelope.py`. The
`StrategyV38Config` dataclass owns the default values; this loader only
overrides them when the TOML provides explicit values.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Literal

from titan.risk.v38_envelope import StrategyV38Config

logger = logging.getLogger(__name__)

_VALID_MODES: tuple[str, ...] = ("shadow", "live")


def load_v38_config_for_strategy(toml_path: str | Path) -> StrategyV38Config:
    """Parse `[v38_envelope]` from a strategy TOML.

    Returns the default `StrategyV38Config(enabled=False, mode="shadow")`
    when the TOML omits the section. Raises `ValueError` on invalid
    `mode` values.

    Parameters
    ----------
    toml_path:
        Path to the strategy's TOML file.

    Returns:
    -------
    `StrategyV38Config` -- frozen dataclass per `v38_envelope.py`.

    Raises:
    ------
    FileNotFoundError
        If `toml_path` does not exist.
    ValueError
        If `[v38_envelope] mode` is set to something other than
        "shadow" or "live".
    """
    path = Path(toml_path)
    if not path.exists():
        raise FileNotFoundError(f"v38 config loader: {toml_path} not found")

    with path.open("rb") as fh:
        data = tomllib.load(fh)

    envelope_table = data.get("v38_envelope")
    if envelope_table is None:
        return StrategyV38Config()

    enabled = bool(envelope_table.get("enabled", False))
    mode_raw = envelope_table.get("mode", "shadow")
    if mode_raw not in _VALID_MODES:
        raise ValueError(
            f"v38 config loader: invalid mode '{mode_raw}' in {toml_path}; "
            f"must be one of {_VALID_MODES}"
        )
    mode: Literal["shadow", "live"] = mode_raw  # type: ignore[assignment]

    return StrategyV38Config(enabled=enabled, mode=mode)


def load_v38_configs_for_dir(
    config_dir: str | Path,
    *,
    strategy_id_from_filename: bool = True,
) -> dict[str, StrategyV38Config]:
    """Scan a directory for TOML files and load per-strategy V3.8 config.

    Returns `dict[strategy_id, StrategyV38Config]` where the strategy_id
    is derived from the filename stem (e.g. `demo_b.toml`
    -> `demo_b`). Strategies with no `[v38_envelope]`
    section get the default disabled config.

    Parameters
    ----------
    config_dir:
        Directory containing strategy TOMLs.
    strategy_id_from_filename:
        When True (default), use the TOML filename stem as the
        strategy_id. When False, use a `[v38_envelope] strategy_id`
        override if present.

    Returns:
    -------
    Dict mapping strategy_id -> StrategyV38Config.

    Raises:
    ------
    FileNotFoundError
        If `config_dir` does not exist.
    """
    dir_path = Path(config_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"v38 config loader: {config_dir} is not a directory")

    out: dict[str, StrategyV38Config] = {}
    for toml_path in sorted(dir_path.glob("*.toml")):
        try:
            cfg = load_v38_config_for_strategy(toml_path)
        except ValueError as e:
            logger.error("v38 config loader: skipping %s: %s", toml_path, e)
            continue
        if strategy_id_from_filename:
            sid = toml_path.stem
        else:
            # Allow explicit override -- useful when multiple TOMLs share a
            # strategy class (e.g. etf_trend_spy / etf_trend_qqq both
            # produce strategy_id "etf_trend" plus a per-instrument suffix).
            with toml_path.open("rb") as fh:
                data = tomllib.load(fh)
            envelope_table = data.get("v38_envelope", {})
            sid = envelope_table.get("strategy_id") or toml_path.stem
        out[sid] = cfg
    return out


__all__ = [
    "load_v38_config_for_strategy",
    "load_v38_configs_for_dir",
]
