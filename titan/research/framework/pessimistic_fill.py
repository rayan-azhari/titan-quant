"""Pessimistic stop-fill stress for the ruin gate (audit P1-15 / H18).

Backtests assume stops fill AT the stop price. In reality a stop fills at the
next available print past it -- on a gap or a fast tape that is materially
worse, so a position meant to cap its loss at 1R loses ~1.5R. A strategy whose
V3.8 ruin gate only passes on optimistic (at-stop) fills is not actually safe.

This module builds a PESSIMISTIC return series -- the worst losing bars (a proxy
for stop-fill events) are worsened by a slippage multiplier (1.5R by default) --
and runs the ruin gate on it. Deployment should require the gate to pass on the
PESSIMISTIC curve, not just the optimistic backtest.

The worst-``loss_quantile`` of losing bars stand in for stop-fill days; for a
strategy with explicit per-trade stops you can instead pass a return series that
already marks the stop-fill bars. Reuses ``assess_strategy_ruin`` (P1-1/P1-2) so
the gate, horizon, and MaxDD level are the doctrine being gated on.
"""

from __future__ import annotations

import pandas as pd

from titan.research.framework.ruin import RuinAssessment, RuinGate, assess_strategy_ruin

DEFAULT_LOSS_MULTIPLIER = 1.5  # a 1R stop fills at ~1.5R (50% slippage past the stop)
DEFAULT_LOSS_QUANTILE = 0.05  # worst 5% of losing bars stand in for stop-fill events


def stress_stop_fills(
    returns: pd.Series,
    *,
    loss_multiplier: float = DEFAULT_LOSS_MULTIPLIER,
    loss_quantile: float = DEFAULT_LOSS_QUANTILE,
) -> pd.Series:
    """Worsen the deepest ``loss_quantile`` of LOSING bars by ``loss_multiplier``.

    Returns a copy of ``returns`` (simple per-bar returns) with the worst losses
    scaled (e.g. a -2% bar in the tail becomes -3% at 1.5x). Non-tail bars and
    all gains are untouched. ``loss_multiplier=1.0`` is a no-op.
    """
    s = returns.dropna().astype(float).copy()
    if loss_multiplier == 1.0 or s.empty:
        return s
    losses = s[s < 0.0]
    if losses.empty:
        return s
    # The loss_quantile-th percentile of the (negative) loss distribution -- a
    # deep negative threshold; bars at or below it are the stop-fill proxies.
    threshold = float(losses.quantile(loss_quantile))
    mask = s <= threshold
    s.loc[mask] = s.loc[mask] * loss_multiplier
    return s


def assess_pessimistic_ruin(
    returns: pd.Series,
    *,
    gate: RuinGate,
    deployment_weight: float = 1.0,
    loss_multiplier: float = DEFAULT_LOSS_MULTIPLIER,
    loss_quantile: float = DEFAULT_LOSS_QUANTILE,
    block_size: int = 21,
    n_paths: int = 1000,
    seed: int = 42,
) -> RuinAssessment:
    """Run the ruin gate on the PESSIMISTIC (stop-slippage-stressed) curve (P1-15).

    Deployment should require ``assess_pessimistic_ruin(...).passes_gate(gate)``
    -- the strategy survives the ruin gate even when stops slip ~1.5R.
    """
    stressed = stress_stop_fills(
        returns, loss_multiplier=loss_multiplier, loss_quantile=loss_quantile
    )
    return assess_strategy_ruin(
        stressed,
        deployment_weight=deployment_weight,
        gate=gate,
        block_size=block_size,
        n_paths=n_paths,
        seed=seed,
    )


__all__ = [
    "DEFAULT_LOSS_MULTIPLIER",
    "DEFAULT_LOSS_QUANTILE",
    "stress_stop_fills",
    "assess_pessimistic_ruin",
]
